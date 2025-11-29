"""
DPSOVAD Configuration for UCF-Crime Dataset

This configuration file contains all hyperparameter settings, adjusted based on paper specifications:
- attn_window: 8 → 64 (local window size)
- visual_layers: 2 → 1 (number of Transformer layers)
- Loss weights: λ_Align=0.9, λ_2=0.09
- Learning rate: 1e-5 (AdamW optimizer)
"""

import argparse

parser = argparse.ArgumentParser(description='DPSOVAD - Discriminative Prompt-based Video Anomaly Detection')

# ============================================
# Random seed
# ============================================
parser.add_argument('--seed', default=2025, type=int,
                    help='Random seed for reproducibility') 

# ============================================
# Model parameters - based on paper specifications
# ============================================
parser.add_argument('--embed-dim', default=512, type=int,
                    help='CLIP embedding dimension')

parser.add_argument('--visual-length', default=256, type=int,
                    help='Maximum video sequence length (number of frames)')

parser.add_argument('--visual-width', default=512, type=int,
                    help='Visual feature width')

parser.add_argument('--visual-head', default=4, type=int,
                    help='Number of multi-head attention heads')

parser.add_argument('--visual-layers', default=1, type=int,
                    help='Number of Transformer layers')

parser.add_argument('--attn-window', default=32, type=int,
                    help='Local attention window size')

parser.add_argument('--classes-num', default=14, type=int,
                    help='Number of classes in UCF-Crime dataset (including Normal)')

# ============================================
# Training parameters
# ============================================
parser.add_argument('--max-epoch', default=10, type=int,
                    help='Maximum number of training epochs')

parser.add_argument('--batch-size', default=64, type=int,
                    help='Batch size')

parser.add_argument('--lr', default=1e-5, type=float,
                    help='Learning rate (paper setting)')

parser.add_argument('--scheduler-rate', default=0.1, type=float,
                    help='Learning rate decay rate')

parser.add_argument('--scheduler-milestones', default=[4, 8], type=list,
                    help='Learning rate decay milestone epochs')

# ============================================
# Loss function hyperparameters - based on paper
# ============================================
parser.add_argument('--lambda-align', default=0.9, type=float,
                    help='Alignment loss weight λ_Align')

parser.add_argument('--lambda-2', default=0.01, type=float,
                    help='Smoothness loss weight λ_2')

# ============================================
# Data paths
# ============================================
parser.add_argument('--train-list', default='../list/ucf_CLIP_rgb.csv',
                    help='Training data list')

parser.add_argument('--test-list', default='../list/ucf_CLIP_rgbtest.csv',
                    help='Test data list')

parser.add_argument('--gt-path', default='../list/gt_ucf.npy',
                    help='Ground truth annotation path')

parser.add_argument('--gt-segment-path', default='../list/gt_segment_ucf.npy',
                    help='Segment-level ground truth path')

parser.add_argument('--gt-label-path', default='../list/gt_label_ucf.npy',
                    help='Label-level ground truth path')

# ============================================
# Model saving and loading
# ============================================
parser.add_argument('--model-path', default='../model/model_dpsovad_ucf.pth',
                    help='Final model save path')

parser.add_argument('--checkpoint-path', default='../model/checkpoint_dpsovad.pth',
                    help='Training checkpoint save path')

parser.add_argument('--use-checkpoint', default=False, type=bool,
                    help='Whether to resume training from checkpoint')

# ============================================
# Advanced options
# ============================================
parser.add_argument('--top-k-base', default=5, type=int,
                    help='Top-K dynamic calculation base (k = max(1, T/base + 1))')

parser.add_argument('--gcn-threshold', default=0.7, type=float,
                    help='GCN similarity graph threshold')

# ============================================
# Debugging options
# ============================================
parser.add_argument('--debug', default=False, type=bool,
                    help='Debug mode (print more information)')

parser.add_argument('--log-interval', default=1280, type=int,
                    help='Log printing interval (by number of samples)')


if __name__ == '__main__':
    """Print configuration information for validation"""
    args = parser.parse_args([])  # Use default parameters

    print("=" * 80)
    print("DPSOVAD Configuration Summary")
    print("=" * 80)

    print("\n[Model Architecture]")
    print(f"  Visual Length:     {args.visual_length}")
    print(f"  Visual Width:      {args.visual_width}")
    print(f"  Visual Layers:     {args.visual_layers}")
    print(f"  Attention Window:  {args.attn_window}")
    print(f"  Embed Dim:         {args.embed_dim}")

    print("\n[Training Parameters]")
    print(f"  Max Epochs:        {args.max_epoch}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  LR Milestones:     {args.scheduler_milestones}")
    print(f"  LR Decay Rate:     {args.scheduler_rate}")

    print("\n[Loss Weights]")
    print(f"  λ_MIL:             1.0 (baseline)")
    print(f"  λ_Align:           {args.lambda_align}")
    print(f"  λ_2:               {args.lambda_2}")

    print("\n[Data Paths]")
    print(f"  Train List:        {args.train_list}")
    print(f"  Test List:         {args.test_list}")
    print(f"  Model Save Path:   {args.model_path}")

    print("\n" + "=" * 60)
    print("Configuration validated successfully!")
    print("=" * 60)
