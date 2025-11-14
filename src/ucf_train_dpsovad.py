"""
DPSOVAD Training Script for UCF-Crime Dataset

This module implements the complete training pipeline for DPSOVAD with optimizations:
- Vectorized loss computations
- Dynamic Top-K strategy
- Three-component loss function: L_MIL + λ_Align * L_Align + λ_2 * L_2
"""

import torch
import warnings
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os

from model_dpsovad import DPSOVAD, dynamic_topk
from ucf_test_dpsovad import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option_dpsovad as ucf_option

# Suppress PyTorch attention mask type mismatch deprecation warning (PyTorch internal issue)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')
# Suppress DPSOVAD text parameter deprecation warning (model always uses memory.txt fixed prompts)
warnings.filterwarnings('ignore', message='.*DPSOVAD.encode_textprompt.*')

def compute_mil_loss(anomaly_scores: torch.Tensor,
                     labels: torch.Tensor,
                     lengths: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """
    L_MIL: 多实例学习损失

    使用Top-K方法获取视频级预测，动态调整k值

    公式：L_MIL = -E[(Y*log(y_video) + (1-Y)*log(1-y_video))]
    其中：y_video = mean(top_k(A))

    Args:
        anomaly_scores: 异常分数 [B, T]
        labels: 视频级标签 [B, C]
        lengths: 实际序列长度 [B]
        device: 计算设备

    Returns:
        MIL损失标量
    """
    instance_logits = torch.zeros(0, device=device)
    labels_video = labels[:, 0]  # 视频级标签

    for i in range(anomaly_scores.shape[0]):
        length = int(lengths[i])
        k_val = dynamic_topk(length)

        # Top-K选择
        tmp, _ = torch.topk(anomaly_scores[i, :length], k=k_val, largest=True)
        video_score = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, video_score], dim=0)

    # 归一化到[0,1]
    instance_logits = torch.sigmoid(instance_logits)

    # 二元交叉熵损失（0=normal, 1=anomaly）
    labels_video = (1 - labels_video).to(device)
    mil_loss = F.binary_cross_entropy(instance_logits, labels_video)

    return mil_loss


def compute_align_loss(logits_pos: torch.Tensor,
                      logits_neg: torch.Tensor,
                      anomaly_scores: torch.Tensor,
                      labels: torch.Tensor,
                      lengths: torch.Tensor,
                      device: torch.device,
                      tau: float = 0.1) -> torch.Tensor:
    """
    L_Align: 对齐损失

    确保视频特征与文本特征的对齐，使用Top-K伪标签生成帧级监督

    流程：
    1. 计算对齐概率 P_i = Softmax([S_pos,i, S_neg,i] / tau)
    2. 生成帧级伪标签：对于异常视频，Top-K片段标记为1，其余为0
    3. 计算交叉熵惩罚预测概率与伪标签之间的差异

    公式：L_Align = -1/T* * sum_i sum_c y_c,i * log(P_c,i)

    Args:
        logits_pos: 正样本相似度 [B, T, 1]
        logits_neg: 负样本相似度 [B, T, 1]
        anomaly_scores: 异常分数 [B, T]
        labels: 视频级标签 [B, C]
        lengths: 实际序列长度 [B]
        device: 计算设备
        tau: 温度系数

    Returns:
        对齐损失标量
    """
    align_loss = torch.tensor(0.0, device=device)

    for i in range(logits_pos.shape[0]):
        length = int(lengths[i])
        label = labels[i, 0]

        # 计算对齐概率
        logits_frame = torch.cat([logits_pos[i, :length], logits_neg[i, :length]], dim=-1)  # [T, 2]
        probs = F.softmax(logits_frame / tau, dim=-1)  # [T, 2]

        # 生成伪标签
        if label == 0:  # 异常视频
            k_val = dynamic_topk(length)
            # 使用detach确保伪标签生成不参与梯度计算
            _, top_indices = torch.topk(anomaly_scores[i, :length].detach(), k=k_val, largest=True)
            pseudo_labels = torch.zeros(length, 2, device=device)
            pseudo_labels[:, 0] = 1.0  # 默认为正常
            pseudo_labels[top_indices, 0] = 0.0  # Top-K标记为异常
            pseudo_labels[top_indices, 1] = 1.0
        else:  # 正常视频
            pseudo_labels = torch.zeros(length, 2, device=device)
            pseudo_labels[:, 0] = 1.0  # 全部标记为正常

        # 交叉熵
        frame_loss = -torch.sum(pseudo_labels * torch.log(probs + 1e-8)) / length
        align_loss += frame_loss

    align_loss = align_loss / logits_pos.shape[0]
    return align_loss


def compute_smoothness_loss(anomaly_scores: torch.Tensor,
                           lengths: torch.Tensor) -> torch.Tensor:
    """
    L_2: 时序平滑损失（向量化版本）

    确保异常分数在时间维度上的平滑性和连贯性
    使用欧几里得距离（L2范数）计算相邻帧之间的差异

    公式：L_2 = 1/(T*-1) * sum_{i=2}^{T*} |A_i - A_{i-1}|^2

    Args:
        anomaly_scores: 异常分数 [B, T]
        lengths: 实际序列长度 [B]

    Returns:
        平滑损失标量
    """
    B, T = anomaly_scores.shape
    device = anomaly_scores.device

    # 构建有效帧mask
    idx = torch.arange(T, device=device).unsqueeze(0)
    valid = idx < lengths.unsqueeze(1)

    # 计算相邻帧差异
    diff = anomaly_scores[:, 1:] - anomaly_scores[:, :-1]  # [B, T-1]
    valid_pair = valid[:, 1:] & valid[:, :-1]  # 两帧都有效才计入

    # 加权L2损失
    diff2 = (diff ** 2) * valid_pair
    per_video = diff2.sum(dim=1) / (lengths - 1).clamp_min(1)

    return per_video.mean()


def train(model: nn.Module,
          normal_loader: DataLoader,
          anomaly_loader: DataLoader,
          testloader: DataLoader,
          args,
          label_map: dict,
          device: torch.device):
    """
    DPSOVAD训练流程

    损失函数：L_Total = L_MIL + λ_Align * L_Align + λ_2 * L_2

    Args:
        model: DPSOVAD模型
        normal_loader: 正常样本数据加载器
        anomaly_loader: 异常样本数据加载器
        testloader: 测试数据加载器
        args: 训练配置
        label_map: 类别映射
        device: 计算设备
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
        loss_total_align = 0
        loss_total_smooth = 0

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
            text_features, logits_pos, logits = model(
                visual_features, None, prompt_text, feat_lengths
            )

            # Get anomaly_scores and logits_neg_max (stored in model attributes)
            anomaly_scores = model.anomaly_scores
            logits_neg_max = model.logits_neg_max

            # Compute three loss components
            loss_mil = compute_mil_loss(anomaly_scores, text_labels, feat_lengths, device)
            loss_align = compute_align_loss(logits_pos, logits_neg_max, anomaly_scores,
                                           text_labels, feat_lengths, device)
            loss_smooth = compute_smoothness_loss(anomaly_scores, feat_lengths)

            # Total loss
            loss_total = loss_mil + lambda_align * loss_align + lambda_2 * loss_smooth

            # Backward propagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            loss_total_mil += loss_mil.item()
            loss_total_align += loss_align.item()
            loss_total_smooth += loss_smooth.item()

            # Periodic evaluation and saving
            step = i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                avg_mil = loss_total_mil / (i + 1)
                avg_align = loss_total_align / (i + 1)
                avg_smooth = loss_total_smooth / (i + 1)

                print(f'Epoch {e+1} | Step {step} | '
                      f'L_MIL: {avg_mil:.4f} | '
                      f'L_Align: {avg_align:.4f} | '
                      f'L_2: {avg_smooth:.4f}')

                # Test performance
                AUC, AP = test(model, testloader, args.visual_length, prompt_text,
                              gt, gtsegments, gtlabels, device)
                AP = AUC

                # Save best model
                if AP > ap_best:
                    ap_best = AP
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best
                    }
                    torch.save(checkpoint, args.checkpoint_path)
                    print(f'Best model saved with AP: {ap_best:.4f}')

        scheduler.step()

        # Save current model at the end of each epoch
        os.makedirs('../model', exist_ok=True)
        torch.save(model.state_dict(), '../model/model_dpsovad_cur.pth')

        # Load best checkpoint
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

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
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    # UCF-Crime dataset label mapping
    label_map = {
        'Normal': 'normal',
        'Abuse': 'abuse',
        'Arrest': 'arrest',
        'Arson': 'arson',
        'Assault': 'assault',
        'Burglary': 'burglary',
        'Explosion': 'explosion',
        'Fighting': 'fighting',
        'RoadAccidents': 'roadAccidents',
        'Robbery': 'robbery',
        'Shooting': 'shooting',
        'Shoplifting': 'shoplifting',
        'Stealing': 'stealing',
        'Vandalism': 'vandalism'
    }

    # Create data loaders
    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
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
        prompt_prefix=args.prompt_prefix,
        prompt_postfix=args.prompt_postfix,
        device=device
    )

    print(f"Starting DPSOVAD training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
