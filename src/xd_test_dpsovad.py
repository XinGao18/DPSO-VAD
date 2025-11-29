"""
XD-Violence Dataset Test Script - DPSOVAD Version

Evaluate DPSOVAD model on XD-Violence test set, computing AUC and AP metrics.
Optimized version: fixes device consistency issues and improves performance.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings

from model_dpsovad import DPSOVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
import xd_option_dpsovad

# Suppress PyTorch attention mask type mismatch deprecation warning (PyTorch internal issue)
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')
# Suppress DPSOVAD text parameter deprecation warning (model always uses memory.txt fixed prompts)
warnings.filterwarnings('ignore', message='.*DPSOVAD.encode_textprompt.*')



def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    """
    Test DPSOVAD model performance on XD-Violence dataset

    Args:
        model: DPSOVAD model
        testdataloader: Test data loader
        maxlen: Maximum video length (segment size)
        prompt_text: Text prompt list
        gt: Ground truth labels
        gtsegments: Ground truth time segments
        gtlabels: Ground truth label names
        device: Compute device

    Returns:
        ROC_AUC: AUC-ROC score
        AP: Average Precision score
    """

    model.to(device)
    model.eval()

    # Use Python list to collect results (more efficient)
    ap_anomaly_list = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            orig_length = int(item[2])
            len_cur = orig_length

            # Short video processing
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            # Compute segment length (keep on CPU for get_batch_mask)
            num_segments = int(orig_length / maxlen) + 1
            lengths = torch.zeros(num_segments, dtype=torch.int64)

            remaining_length = orig_length
            for j in range(num_segments):
                if remaining_length >= maxlen:
                    lengths[j] = maxlen
                    remaining_length -= maxlen
                else:
                    lengths[j] = remaining_length

            # Keep lengths on CPU for get_batch_mask
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            # Move lengths to device for model use
            lengths_device = lengths.to(device)

            # DPSOVAD forward propagation
            text_features, logits_pos, logits = model(visual, padding_mask, prompt_text, lengths_device)

            # Get anomaly scores (from model attributes)
            anomaly_scores = model.anomaly_scores  # [B, T]

            # Reshape
            anomaly_scores_flat = anomaly_scores.reshape(-1)

            # Extract anomaly scores for valid length
            prob_anomaly = anomaly_scores_flat[0:len_cur]

            # Accumulate to list (avoid repeated cat operations)
            ap_anomaly_list.append(prob_anomaly.cpu())

    # Merge anomaly scores from all videos
    ap_anomaly = torch.cat(ap_anomaly_list, dim=0).numpy()

    # Evaluate using anomaly scores
    if len(ap_anomaly) == 0:
        raise ValueError("ap_anomaly is empty, cannot calculate evaluation metrics.")

    # Dynamic frame-level upsampling with length alignment
    seg_len = len(ap_anomaly)
    gt_len = len(gt)

    repeat_factor = int(np.round(gt_len / seg_len))
    frame_scores = np.repeat(ap_anomaly, repeat_factor)

    ROC_anomaly = roc_auc_score(gt, frame_scores)
    AP_anomaly = average_precision_score(gt, frame_scores)
    return ROC_anomaly, AP_anomaly


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    args = xd_option_dpsovad.parser.parse_args()

    # XD-Violence label mapping
    label_map = dict({
        'A': 'normal',
        'B1': 'fighting',
        'B2': 'shooting',
        'B4': 'riot',
        'B5': 'abuse',
        'B6': 'car accident',
        'G': 'explosion'
    })

    # Load test data
    print("Loading test data...")
    testdataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    # Get prompt text
    prompt_text = get_prompt_text(label_map)
    print(f"Number of prompts: {len(prompt_text)}")

    # Load ground truth
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)
    print(f"Ground truth loaded (total frames: {len(gt)})")

    # Create DPSOVAD model
    print("\nCreating DPSOVAD model...")
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

    # Load trained model weights
    print(f"Loading model weights: {args.model_path}")
    model_param = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(model_param)
    print("Model loaded successfully")

    # Run test
    print("\nStarting test...")
    auc, ap = test(
        model, testdataloader, args.visual_length, prompt_text,
        gt, gtsegments, gtlabels, device
    )

    print(f"\nFinal results: AUC={auc:.4f}, AP={ap:.4f}")
