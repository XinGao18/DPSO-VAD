# DPSOVAD: Discriminative Prompt-based Video Anomaly Detection

PyTorch implementation of **DPSOVAD (Discriminative Prompt-based Spatio-temporal Video Anomaly Detection)** for weakly supervised video anomaly detection.

## Overview

DPSOVAD is an advanced framework for video anomaly detection that leverages vision-language models with discriminative prompts. The model introduces fixed positive and negative text prompts to enhance anomaly discrimination through contrastive learning.

### Key Features

- **Fixed Discriminative Prompts**: Uses pre-defined positive and negative prompt pairs from `memory.txt` for consistent and robust anomaly detection
- **LGT-Adapter**: Local-Global Temporal adapter that captures temporal dependencies through:
  - Local module: Windowed Transformer for local temporal patterns
  - Global module: Graph Convolutional Networks for long-range dependencies
- **Three-Component Loss Function**:
  - L_MIL: Multiple Instance Learning loss for video-level supervision
  - L_Align: Alignment loss for visual-text feature matching
  - L_2: Temporal smoothness loss for coherent predictions
- **Dynamic Top-K Strategy**: Adaptive selection of top anomalous snippets based on video length

### Architecture Highlights

- **visual_head**: 2 (multi-head attention heads)
- **visual_layers**: 1 (Transformer layer)
- **attn_window**: 64 (local attention window size)
- **Loss weights**: λ_Align=0.9, λ_2=0.09

## Performance

Performance on standard benchmarks:

| Dataset | AUC | AP |
|---------|-----|-----|
| UCF-Crime | TBD | TBD |
| XD-Violence | TBD | TBD |

## Requirements

```bash
torch>=1.12.0
numpy
pandas
scikit-learn
clip  # OpenAI CLIP
```

## Installation

```bash
git clone https://github.com/your-repo/VadCLIP.git
cd VadCLIP
pip install -r requirements.txt
```

## Data Preparation

### Pre-extracted CLIP Features

Download pre-extracted CLIP features from:
- **Baidu Cloud**: [Link TBD]
- **OneDrive**: [Link TBD]

The features should be organized as follows:
```
data/
├── UCFTrainClipFeatures/
├── UCFTestClipFeatures/
├── XDTrainClipFeatures/
└── XDTestClipFeatures/
```

### Update Data Paths

Update the file paths in the CSV files to point to your local data directory:

**For XD-Violence dataset:**
```bash
# Update paths in list/xd_CLIP_rgb.csv and list/xd_CLIP_rgbtest.csv
# Example: Replace /home/xbgydx/Desktop with ../data or your local path
sed -i 's|/home/xbgydx/Desktop|../data|g' list/xd_CLIP_rgb.csv
sed -i 's|/home/xbgydx/Desktop|../data|g' list/xd_CLIP_rgbtest.csv
```

**For UCF-Crime dataset:**
```bash
# Update paths in list/ucf_CLIP_rgb.csv and list/ucf_CLIP_rgbtest.csv
sed -i 's|/home/xbgydx/Desktop|../data|g' list/ucf_CLIP_rgb.csv
sed -i 's|/home/xbgydx/Desktop|../data|g' list/ucf_CLIP_rgbtest.csv
```

## Training

### UCF-Crime Dataset

```bash
cd src
python ucf_train_dpsovad.py
```

Configuration options can be modified in `src/ucf_option_dpsovad.py`:
- `--visual-head 2`: Number of attention heads
- `--visual-layers 1`: Number of Transformer layers
- `--attn-window 64`: Local attention window size
- `--batch-size 64`: Training batch size
- `--lr 1e-5`: Learning rate
- `--lambda-align 0.9`: Alignment loss weight
- `--lambda-2 0.09`: Smoothness loss weight

### XD-Violence Dataset

```bash
cd src
python xd_train_dpsovad.py
```

Configuration options can be modified in `src/xd_option_dpsovad.py`:
- `--batch-size 96`: Training batch size
- `--classes-num 7`: Number of classes (Normal + 6 anomaly types)

## Testing

### UCF-Crime Dataset

```bash
cd src
python ucf_test_dpsovad.py
```

### XD-Violence Dataset

```bash
cd src
python xd_test_dpsovad.py
```

## Fixed Text Prompts

DPSOVAD uses fixed prompts defined in `memory.txt`:

**Positive Prompts** (normal activity):
```
<Positive Prompts>
a photo of action
a video of action
```

**Negative Prompts** (anomalous activity):
```
<Negative Prompts>
a photo of inaction
a video of inaction
```

These prompts are automatically loaded by the model and should not be modified during inference.

## Model Files

### Pre-trained Models

Download trained models from:
- **Baidu Cloud**: [Link TBD]
- **OneDrive**: [Link TBD]

Place the model files in the `model/` directory:
```
model/
├── model_dpsovad_ucf.pth
└── model_dpsovad_xd.pth
```

### Training from Scratch

Models will be saved to:
- Best checkpoint: `model/checkpoint_dpsovad_[ucf/xd].pth`
- Current model: `model/model_dpsovad_[ucf/xd]_cur.pth`
- Final model: `model/model_dpsovad_[ucf/xd].pth`

## Project Structure

```
VadCLIP/
├── src/
│   ├── model_dpsovad.py           # DPSOVAD model architecture
│   ├── ucf_train_dpsovad.py       # UCF-Crime training script
│   ├── ucf_test_dpsovad.py        # UCF-Crime testing script
│   ├── ucf_option_dpsovad.py      # UCF-Crime configuration
│   ├── xd_train_dpsovad.py        # XD-Violence training script
│   ├── xd_test_dpsovad.py         # XD-Violence testing script
│   ├── xd_option_dpsovad.py       # XD-Violence configuration
│   └── utils/
│       ├── dataset.py             # Dataset loaders
│       └── tools.py               # Utility functions
├── list/
│   ├── ucf_CLIP_rgb.csv           # UCF-Crime training list
│   ├── ucf_CLIP_rgbtest.csv       # UCF-Crime test list
│   ├── xd_CLIP_rgb.csv            # XD-Violence training list
│   ├── xd_CLIP_rgbtest.csv        # XD-Violence test list
│   ├── gt_ucf.npy                 # UCF-Crime ground truth
│   └── gt.npy                     # XD-Violence ground truth
├── model/                         # Saved models directory
├── data/                          # CLIP features directory
├── memory.txt                     # Fixed text prompts
└── README.md
```

## Key Implementation Details

### DPSOVAD Model Architecture

1. **Frozen CLIP Encoders**: Uses pre-trained ViT-B/16 CLIP model with frozen parameters
2. **LGT-Adapter Components**:
   - Windowed Transformer with local attention (window size: 64)
   - Similarity-based GCN with cosine similarity adjacency
   - Distance-based GCN with exponential distance adjacency
3. **Visual Prompt Fusion**: MLP-based fusion of visual and text features
4. **Anomaly Score**: A = S_neg - S_pos (difference between negative and positive similarities)

### Training Process

1. **Data Loading**: Separate loaders for normal and anomaly videos
2. **Batch Composition**: Each batch contains equal normal and anomaly samples
3. **Loss Computation**:
   - L_MIL: Top-K based video-level prediction
   - L_Align: Frame-level pseudo-label alignment
   - L_2: Temporal smoothness regularization
4. **Optimization**: AdamW optimizer with multi-step learning rate decay
5. **Evaluation**: Periodic testing with checkpoint saving based on AP metric

### Dynamic Top-K Strategy

```python
k = max(1, T / 16 + 1)
```
where T is the video length (number of segments).

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{dpsovad2024,
  title={DPSOVAD: Discriminative Prompt-based Spatio-temporal Video Anomaly Detection},
  author={Xin Gao},
  booktitle={I do not know},
  year={2025}
}
```

## Acknowledgments

This implementation builds upon:
- [VadCLIP](https://github.com/nwpu-zxr/VadCLIP): Original VadCLIP implementation
- [OpenAI CLIP](https://github.com/openai/CLIP): Pre-trained vision-language models
- [XDVioDet](https://github.com/Roc-Ng/XDVioDet): XD-Violence dataset and baseline
- [DeepMIL](https://github.com/Roc-Ng/DeepMIL): Multiple instance learning baseline

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [gxhh521@163.com].
