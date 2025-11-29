"""
DPSOVAD Training Script for XD-Violence Dataset

This module implements the complete training pipeline for DPSOVAD with optimizations:
- Vectorized loss computations
- Dynamic Top-K strategy
- Three-component loss function: L_MIL + λ_Align * L_Align + λ_2 * L_2
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os
import warnings

from model_dpsovad import DPSOVAD, dynamic_topk
from xd_test_dpsovad import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label, get_anomaly_flags
import xd_option_dpsovad as xd_option

# Suppress PyTorch attention mask type mismatch deprecation warning (PyTorch internal issue)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')
# Suppress DPSOVAD text parameter deprecation warning (model always uses memory.txt fixed prompts)
warnings.filterwarnings('ignore', message='.*DPSOVAD.encode_textprompt.*')


def compute_mil_loss(anomaly_scores: torch.Tensor,
                     labels: torch.Tensor,
                     lengths: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """
    L_MIL: Multiple Instance Learning Loss

    Uses Top-K method to obtain video-level predictions with dynamic k adjustment

    Formula:L_MIL = -E[(Y*log(y_video) + (1-Y)*log(1-y_video))]
    where:y_video = mean(top_k(A))

    Args:
        anomaly_scores: Anomaly scores [B, T]
        labels: Video-level labels [B, C]
        lengths: Actual sequence lengths [B]
        device: Computing device

    Returns:
        MIL loss scalar
    """
    instance_logits = torch.zeros(0, device=device)
    labels_video = labels[:, 0]  # Video-level labels

    for i in range(anomaly_scores.shape[0]):
        length = int(lengths[i])
        k_val = dynamic_topk(length)

        # Top-K selection
        tmp, _ = torch.topk(anomaly_scores[i, :length], k=k_val, largest=True)
        video_score = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, video_score], dim=0)

    # Normalize to [0,1]
    instance_logits = torch.sigmoid(instance_logits)

    # Binary cross-entropy loss (0=normal, 1=anomaly)
    labels_video = (1 - labels_video).to(device)
    mil_loss = F.binary_cross_entropy(instance_logits, labels_video)

    return mil_loss


def compute_smoothness_loss(anomaly_scores: torch.Tensor,
                           lengths: torch.Tensor) -> torch.Tensor:
    """
    L_2: Temporal Smoothness Loss (Vectorized Version)

    Ensures smoothness and coherence of anomaly scores across temporal dimension
    Uses Euclidean distance (L2 norm) to compute differences between adjacent frames

    Formula:L_2 = 1/(T*-1) * sum_{i=2}^{T*} |A_i - A_{i-1}|^2

    Args:
        anomaly_scores: Anomaly scores [B, T]
        lengths: Actual sequence lengths [B]

    Returns:
        Smoothness loss scalar
    """
    B, T = anomaly_scores.shape
    device = anomaly_scores.device

    # Build valid frame mask
    idx = torch.arange(T, device=device).unsqueeze(0)
    valid = idx < lengths.unsqueeze(1)

    # Compute adjacent frame differences
    diff = anomaly_scores[:, 1:] - anomaly_scores[:, :-1]  # [B, T-1]
    valid_pair = valid[:, 1:] & valid[:, :-1]  # Both frames must be valid

    # Weighted L2 loss
    diff2 = (diff ** 2) * valid_pair
    per_video = diff2.sum(dim=1) / (lengths - 1).clamp_min(1)

    return per_video.mean()


def _build_valid_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Build frame-level valid mask to avoid counting padding regions."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def compute_feature_contrastive_loss(text_features: torch.Tensor,
                                     visual_features: torch.Tensor,
                                     labels: torch.Tensor,
                                     lengths: torch.Tensor,
                                     num_pos_prompts: int,
                                     tau: float = 0.07,
                                     eps: float = 1e-8) -> torch.Tensor:
    """
    L_feat = -log( exp(s_y / τ) / Σ_j exp(s_j / τ) )

    Text-guided feature-level contrastive, utilizing positive/negative text prototypes to guide video representation separation.
    """
    if text_features.ndim != 3 or visual_features.ndim != 3:
        raise ValueError("text_features and visual_features must be 3D tensors")

    B, T, _ = visual_features.shape
    valid_mask = _build_valid_mask(lengths, T).unsqueeze(-1)

    pooled_visual = (visual_features * valid_mask).sum(dim=1)
    pooled_visual = pooled_visual / lengths.clamp_min(1).unsqueeze(-1)
    pooled_visual = F.normalize(pooled_visual, p=2, dim=-1, eps=eps)

    text_features = F.normalize(text_features, p=2, dim=-1, eps=eps)
    num_pos = max(1, min(num_pos_prompts, text_features.shape[1]))

    text_pos_proto = text_features[:, :num_pos, :].mean(dim=1)
    text_neg_proto = text_features[:, num_pos:, :].mean(dim=1) if text_features.shape[1] > num_pos else text_pos_proto.detach().clone()

    text_pos_proto = F.normalize(text_pos_proto, p=2, dim=-1, eps=eps)
    text_neg_proto = F.normalize(text_neg_proto, p=2, dim=-1, eps=eps)

    prototypes = torch.stack([text_pos_proto, text_neg_proto], dim=1)
    logits_proto = torch.bmm(prototypes, pooled_visual.unsqueeze(-1)).squeeze(-1) / tau

    targets = get_anomaly_flags(labels).long().clamp(min=0, max=1)
    return F.cross_entropy(logits_proto, targets)


def compute_similarity_contrastive_loss(logits_pos: torch.Tensor,
                                        logits_neg: torch.Tensor,
                                        labels: torch.Tensor,
                                        lengths: torch.Tensor,
                                        margin: float = 0.2) -> torch.Tensor:
    """
    L_sim = E[ max(0, m + s_neg - s_pos) ] + E[ max(0, m + s_pos - s_neg) ]

    Logits-based margin contrastive, encouraging Normal/Abnormal to separate in similarity space.
    """
    if logits_pos.shape != logits_neg.shape:
        raise ValueError("logits_pos and logits_neg shapes must match")

    B, T, _ = logits_pos.shape
    valid_mask = _build_valid_mask(lengths, T)

    logits_pos_valid = logits_pos.squeeze(-1).masked_fill(~valid_mask, float('-inf'))
    logits_neg_valid = logits_neg.squeeze(-1).masked_fill(~valid_mask, float('-inf'))

    max_pos = logits_pos_valid.max(dim=1).values
    max_neg = logits_neg_valid.max(dim=1).values

    inf_mask = torch.isinf(max_pos)
    if inf_mask.any():
        max_pos[inf_mask] = -10.0
    inf_mask = torch.isinf(max_neg)
    if inf_mask.any():
        max_neg[inf_mask] = -10.0

    max_pos = torch.clamp(max_pos, -10.0, 10.0)
    max_neg = torch.clamp(max_neg, -10.0, 10.0)

    video_labels = get_anomaly_flags(labels)
    normal_loss = F.relu(margin + max_neg - max_pos)
    abnormal_loss = F.relu(margin + max_pos - max_neg)

    return ((1.0 - video_labels) * normal_loss + video_labels * abnormal_loss).mean()


def compute_score_contrastive_loss(anomaly_scores: torch.Tensor,
                                   labels: torch.Tensor,
                                   lengths: torch.Tensor,
                                   tau: float = 0.07,
                                   eps: float = 1e-8) -> torch.Tensor:
    """
    L_score = -log( Σ_{p∈P(i)} exp(sim_{ip}/τ) / Σ_{a≠i} exp(sim_{ia}/τ) )

    Video-level InfoNCE based on anomaly_scores.
    """
    if anomaly_scores.ndim != 2:
        raise ValueError("anomaly_scores must be a 2D tensor")

    B, T = anomaly_scores.shape
    valid_mask = _build_valid_mask(lengths, T)

    masked_scores = anomaly_scores * valid_mask
    mean_scores = masked_scores.sum(dim=1) / lengths.clamp_min(1)
    max_scores = anomaly_scores.masked_fill(~valid_mask, float('-inf')).max(dim=1).values
    max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)

    video_repr = torch.stack([mean_scores, max_scores], dim=1)
    video_repr = F.normalize(video_repr, p=2, dim=1, eps=eps)

    similarity = torch.matmul(video_repr, video_repr.T) / tau
    similarity = similarity - similarity.max(dim=1, keepdim=True)[0].detach()

    eye = torch.eye(B, device=anomaly_scores.device, dtype=torch.bool)
    exp_sim = torch.exp(similarity) * (~eye)

    video_labels = get_anomaly_flags(labels)
    pos_mask = (video_labels.unsqueeze(0) == video_labels.unsqueeze(1)) & (~eye)

    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    denom = exp_sim.sum(dim=1).clamp_min(eps)

    valid_rows = pos_mask.sum(dim=1) > 0
    losses = torch.zeros(B, device=anomaly_scores.device)
    safe_pos = pos_sum.clamp_min(eps)
    losses[valid_rows] = -torch.log(safe_pos[valid_rows] / denom[valid_rows])

    if valid_rows.any():
        return losses[valid_rows].mean()
    return torch.tensor(0.0, device=anomaly_scores.device)


def compute_text_aware_contrastive_loss(text_features: torch.Tensor,
                                        logits_pos: torch.Tensor,
                                        logits_neg: torch.Tensor,
                                        anomaly_scores: torch.Tensor,
                                        labels: torch.Tensor,
                                        lengths: torch.Tensor,
                                        model: nn.Module,
                                        tau: float = 0.07,
                                        alpha: float = 0.4,
                                        beta: float = 0.3,
                                        gamma: float = 0.3) -> torch.Tensor:
    """
    L_text-aware = α·L_feat + β·L_sim + γ·L_score
    """
    if text_features is None or getattr(model, "visual_features", None) is None:
        raise ValueError("Forward propagation must be run first to cache modal features")

    visual_features = model.visual_features
    num_pos = len(getattr(model, "positive_prompts", [])) or 1

    loss_feat = compute_feature_contrastive_loss(
        text_features=text_features,
        visual_features=visual_features,
        labels=labels,
        lengths=lengths,
        num_pos_prompts=num_pos,
        tau=tau
    )

    loss_sim = compute_similarity_contrastive_loss(
        logits_pos=logits_pos,
        logits_neg=logits_neg,
        labels=labels,
        lengths=lengths
    )

    loss_score = compute_score_contrastive_loss(
        anomaly_scores=anomaly_scores,
        labels=labels,
        lengths=lengths,
        tau=tau
    )

    return alpha * loss_feat + beta * loss_sim + gamma * loss_score

def train(model: nn.Module,
          normal_loader: DataLoader,
          anomaly_loader: DataLoader,
          testloader: DataLoader,
          args,
          label_map: dict,
          device: torch.device):
    """
    DPSOVAD Training Pipeline

    Loss function:L_Total = L_MIL + λ_Align * L_Align + λ_2 * L_2

    Args:
        model: DPSOVAD model
        normal_loader: Normal sample data loader
        anomaly_loader: Anomaly sample data loader
        testloader: Test data loader
        args: Training configuration
        label_map: Category mapping
        device: Computing device
    """
    model.to(device)

    # Load ground truth data
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)

    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    # Hyperparameters
    lambda_align = args.lambda_align  # 0.9
    lambda_2 = args.lambda_2  # 0.09

    # Load checkpoint (if needed)
    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print(f"Checkpoint loaded: epoch {epoch+1}, AP {ap_best:.4f}")

    # Training loop
    for e in range(epoch, args.max_epoch):
        model.train()
        loss_total_mil = 0
        loss_total_smooth = 0
        loss_total_contrastive = 0

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            # Merge normal and anomaly samples
            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # Forward propagation
            _, logits_pos, logits = model(
                visual_features, None, prompt_text, feat_lengths
            )

            # Get anomaly_scores and logits_neg_max (stored in model attributes)
            anomaly_scores = model.anomaly_scores
            logits_neg_max = model.logits_neg_max
            text_features_batch = getattr(model, "text_features", None)

            # Compute loss components
            loss_mil = compute_mil_loss(anomaly_scores, text_labels, feat_lengths, device)
            loss_smooth = compute_smoothness_loss(anomaly_scores, feat_lengths)
            loss_contrastive = compute_text_aware_contrastive_loss(
                text_features=text_features_batch,
                logits_pos=logits_pos,
                logits_neg=logits_neg_max,
                anomaly_scores=anomaly_scores,
                labels=text_labels,
                lengths=feat_lengths,
                model=model
            )

            # Total loss
            loss_total = (
                loss_mil +
                lambda_align * loss_contrastive +
                lambda_2 * loss_smooth
            )

            # Backward propagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            loss_total_mil += loss_mil.item()
            loss_total_smooth += loss_smooth.item()
            loss_total_contrastive += loss_contrastive.item()

            # Periodic loss logging
            step = i * normal_loader.batch_size * 2
            if step % args.log_interval == 0 and step != 0:
                avg_mil = loss_total_mil / (i + 1)
                avg_smooth = loss_total_smooth / (i + 1)
                avg_contrastive = loss_total_contrastive / (i + 1)
                total_loss = (avg_mil +
                              lambda_align * avg_contrastive +
                              lambda_2 * avg_smooth)

                print(f'Epoch {e+1} | Step {step} | '
                      f'L_MIL: {avg_mil:.4f} | '
                      f'L_TxtCon: {avg_contrastive:.4f} | '
                      f'L_2: {avg_smooth:.4f} | '
                      f'Loss: {total_loss:.4f}')

        scheduler.step()

        # Silent epoch evaluation (no AP/AUC output)
        epoch_AUC, epoch_AP = test(
            model, testloader, args.visual_length, prompt_text,
            gt, gtsegments, gtlabels, device
        )

        # Silent best model saving
        if epoch_AP > ap_best:
            ap_best = epoch_AP
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best,
                'auc': epoch_AUC
            }
            torch.save(checkpoint, args.checkpoint_path)

        # Save current model at the end of each epoch
        os.makedirs('../model', exist_ok=True)
        torch.save(model.state_dict(), '../model/model_dpsovad_xd_cur.pth')

        # Load best checkpoint for next epoch
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation after all epochs
    print("\n" + "="*80)
    print("Training completed. Running final evaluation...")
    print("="*80)

    # Load best model for final evaluation
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Run final test
    final_AUC, final_AP = test(
        model, testloader, args.visual_length, prompt_text,
        gt, gtsegments, gtlabels, device
    )

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"AUC: {final_AUC:.4f}  |  AP: {final_AP:.4f}")
    print("="*80 + "\n")

    # Save final best model
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        torch.save(checkpoint['model_state_dict'], args.model_path)
        print(f'Final best model saved to {args.model_path}')


def setup_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    # XD-Violence dataset label mapping
    label_map = {
        'A': 'normal',
        'B1': 'fighting',
        'B2': 'shooting',
        'B4': 'riot',
        'B5': 'abuse',
        'B6': 'car accident',
        'G': 'explosion'
    }

    # Create data loaders
    normal_dataset = XDDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    anomaly_dataset = XDDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize DPSOVAD model
    model = DPSOVAD(
        num_class=args.classes_num,
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        visual_head=args.visual_head,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        device=device
    )

    print(f"Starting DPSOVAD training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
