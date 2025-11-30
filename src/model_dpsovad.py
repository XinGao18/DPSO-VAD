"""
DPSOVAD: Discriminative Prompt-based video anomaly detection with Spatio-temporal awareness

This module implements the complete DPSOVAD framework with optimizations based on code review.

Key improvements:
- L2 normalization with epsilon to prevent NaN
- Robust attention mask construction
- Device management based on input tensors
- Vectorized GCN adjacency computation
"""

from collections import OrderedDict
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    L2 normalization with epsilon for numerical stability

    Args:
        x: Input tensor
        dim: Dimension to normalize
        eps: Small constant to prevent division by zero

    Returns:
        Normalized tensor
    """
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def dynamic_topk(length: int, base: int = 16, min_k: int = 1) -> int:
    """
    Dynamic Top-K calculation for multiple instance learning

    Args:
        length: Sequence length
        base: Base value for division (default: 16)
        min_k: Minimum k value (default: 1)

    Returns:
        Computed k value
    """
    return max(min_k, int(length / base + 1))


class LayerNorm(nn.LayerNorm):
    """Custom LayerNorm for mixed precision training"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Fast GELU activation"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Residual attention block with local window attention"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    """Transformer encoder for local temporal modeling"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module

    This module uses multi-head cross-attention to let text prompt vectors (Query) adaptively retrieve visual frame sequences (Key/Value),
    focusing on the key frames that best explain the current semantics, thus achieving robust cross-modal semantic alignment. Internally, it uses residual +
    LayerNorm structure with Dropout, balancing training stability and generalization.

    Args:
        embed_dim (int): Feature dimension, default 512.
        num_heads (int): Number of attention heads, default 8.
        dropout (float): Dropout rate, default 0.1.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Text-visual cross-attention fusion.

        Args:
            text_features (torch.Tensor): Text Query, shape [B, C, 512].
            visual_features (torch.Tensor): Visual Key/Value, shape [B, T, 512].

        Returns:
            torch.Tensor: Aligned text features, shape [B, C, 512].
        """
        # Let text Query aggregate key frame information from visual Key/Value according to semantic needs
        fused, _ = self.cross_attn(
            query=text_features,
            key=visual_features,
            value=visual_features,
            need_weights=False,
        )  # fused: [B, C, 512]
        fused = self.dropout(fused)
        # Residual keeps original semantics, LayerNorm stabilizes training
        return self.norm(text_features + fused)


class DPSOVAD(nn.Module):
    """
    DPSOVAD model: Discriminative Prompt-based Video Anomaly Detection

    Key components:
    1. Frozen CLIP visual/text encoders
    2. Local-Global Temporal (LGT) Adapter
    3. Visual prompt fusion module
    """
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.device = device

        # Local module: Transformer with windowed attention
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        # Global module: Graph Convolutional Networks
        width = int(visual_width / 2)
        # Similarity branch
        self.gc1 = GraphConvolution(visual_width, width, residual=True)  # 512 -> 256
        self.gc2 = GraphConvolution(width, width, residual=True)  # 256 -> 256
        # Distance branch
        self.gc3 = GraphConvolution(visual_width, width, residual=True)  # 512 -> 256
        self.gc4 = GraphConvolution(width, width, residual=True)  # 256 -> 256

        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        # Visual prompt fusion modules
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.cross_fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=8, dropout=0.1)
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))

        # Frozen CLIP pre-trained model
        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        # Learnable embeddings
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        # Load fixed prompts from memory.txt
        self.positive_prompts, self.negative_prompts = self._load_prompts_from_memory()

        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize learnable parameters"""
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def _load_prompts_from_memory(self):
        """
        Load fixed positive and negative prompt pairs from memory.txt

        Returns:
            positive_prompts: List of positive prompt strings
            negative_prompts: List of negative prompt strings
        """
        from pathlib import Path

        # Use pathlib to get memory.txt from project root
        project_root = Path(__file__).resolve().parent.parent
        memory_path = project_root / "memory.txt"

        positive_prompts = []
        negative_prompts = []
        current_section = None

        try:
            with memory_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if "<Positive Prompts>" in line:
                        current_section = "positive"
                    elif "<Negative Prompts>" in line:
                        current_section = "negative"
                    elif current_section == "positive":
                        positive_prompts.append(line)
                    elif current_section == "negative":
                        negative_prompts.append(line)

            # Validate successful loading
            if not positive_prompts or not negative_prompts:
                raise ValueError(
                    f"memory.txt at {memory_path} must contain both positive and negative prompts."
                )

            print(f"Loaded {len(positive_prompts)} positive prompts and "
                  f"{len(negative_prompts)} negative prompts from {memory_path}")
            return positive_prompts, negative_prompts

        except (FileNotFoundError, UnicodeDecodeError, PermissionError, OSError, ValueError) as e:
            print(f"Warning: {e}. Falling back to default prompts.")
            # Use default prompts as fallback
            return (
                [
                    "There is a people are walking on the road.",
                    "There is a normal scence without any adnormal.",
                ],
                [
                    "There is a people are lying on the road.",
                    "There is a anomaly scence in the frame cause something not usual appear in that time or place.",
                ],
            )

    def build_attention_mask(self, attn_window: int) -> torch.Tensor:
        """
        Build robust local window attention mask

        This version handles all edge cases including:
        - visual_length < attn_window
        - visual_length not divisible by attn_window

        Args:
            attn_window: Window size for local attention

        Returns:
            Attention mask tensor [visual_length, visual_length]
        """
        T = self.visual_length
        attn_window = max(1, int(attn_window))
        mask = torch.full((T, T), float('-inf'))

        for start in range(0, T, attn_window):
            end = min(start + attn_window, T)
            mask[start:end, start:end] = 0.0

        return mask

    def compute_similarity_adj(self, x: torch.Tensor, seq_len: torch.Tensor = None):
        """
        Compute similarity-based adjacency matrix (vectorized version)

        Args:
            x: Feature tensor [B, T, D]
            seq_len: Actual sequence lengths [B]

        Returns:
            Adjacency matrix [B, T, T]
        """
        eps = 1e-6
        device = x.device

        # Normalize features
        x_norm = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
        # Compute cosine similarity
        sim = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, T, T]

        # Apply threshold
        sim = F.threshold(sim, 0.7, 0.0)

        # Mask padding frames if lengths provided
        if seq_len is not None:
            # Ensure seq_len is on the same device
            seq_len = seq_len.to(device)
            B, T, _ = sim.shape
            idx = torch.arange(T, device=device).unsqueeze(0)
            valid = idx < seq_len.unsqueeze(1)
            # Only apply softmax between valid frames
            mask = (~valid).unsqueeze(1) | (~valid).unsqueeze(2)
            sim = sim.masked_fill(mask, float('-inf'))

        adj = F.softmax(sim, dim=-1).nan_to_num(0.0)
        return adj

    def encode_video(self, images, padding_mask, lengths):
        """
        Encode video through LGT adapter

        Flow:
        1. Add position embeddings
        2. Local module: Windowed Transformer
        3. Global module: Similarity GCN + Distance GCN
        4. Feature fusion

        Args:
            images: Video features [B, T, D]
            padding_mask: Padding mask
            lengths: Actual sequence lengths [B]

        Returns:
            Enhanced video features [B, T, D]
        """
        images = images.to(torch.float)
        device = images.device

        # Ensure lengths is on the same device if provided
        if lengths is not None:
            lengths = lengths.to(device)

        # Add frame position embeddings
        position_ids = torch.arange(self.visual_length, device=device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        # Construct padding mask if needed
        if lengths is not None and padding_mask is None:
            max_T = images.shape[0]  # Time is the first dim after permute
            idx = torch.arange(max_T, device=device).unsqueeze(1)
            padding_mask = idx >= lengths.unsqueeze(0)
            padding_mask = padding_mask.transpose(0, 1)  # [B, T]

        # Local module: Capture local temporal dependencies
        x, _ = self.temporal((images, padding_mask))
        x = x.permute(1, 0, 2)  # [B, T, D]

        # Global module: Construct similarity and distance adjacency matrices
        adj_sim = self.compute_similarity_adj(x, lengths)
        adj_dis = self.disAdj(x.shape[0], x.shape[1], device=device, lengths=lengths)

        # Similarity branch
        x1_h = self.gelu(self.gc1(x, adj_sim))
        x1 = self.gelu(self.gc2(x1_h, adj_sim))

        # Distance branch
        x2_h = self.gelu(self.gc3(x, adj_dis))
        x2 = self.gelu(self.gc4(x2_h, adj_dis))

        # Fuse two branches
        x = torch.cat((x1, x2), 2)  # [B, T, 512]
        x = self.linear(x)  # [B, T, 512]

        return x

    def encode_textprompt(self, text=None):
        """
        Encode video using fixed prompts

        The text parameter is kept for interface compatibility but will be ignored;
        the model always uses fixed prompts loaded from memory.txt.

        Args:
            text: List of text prompts (deprecated, will be ignored)

        Returns:
            Text features [N, D] where N is number of prompts
        """
        if text is not None:
            warnings.warn(
                "The text parameter in DPSOVAD.encode_textprompt(text) is deprecated and will be ignored; "
                "the model always uses fixed prompts defined in memory.txt.",
                UserWarning,
            )

        # Fixed positive and negative prompts
        all_prompts = self.positive_prompts + self.negative_prompts

        # Dynamically get the device CLIP is on, avoiding dependency on self.device
        device = next(self.clipmodel.parameters()).device

        # 1) tokenize -> 2) token embedding -> 3) encode_text
        text_tokens = clip.tokenize(all_prompts).to(device)
        text_embeddings = self.clipmodel.encode_token(text_tokens)
        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features  # [N_prompts, D]

    def forward(self, visual, padding_mask, text, lengths):
        """
        Forward propagation

        Args:
            visual: Video features [B, T, D]
            padding_mask: Padding mask
            text: List of text prompts
            lengths: Actual sequence lengths [B]

        Returns:
            text_features_ori: Original text features [C, D] (for compatibility)
            logits_pos: Positive similarity [B, T, 1] (used for MIL/Align loss)
            logits: Full similarity logits [B, T, C] (used for test evaluation)

        Note:
            During training, also computes:
            - logits_neg_max: Max negative similarity [B, T, 1]
            - anomaly_scores: Anomaly score A = S_neg - S_pos [B, T]
            These are stored as attributes for loss computation.
        """
        # Encode video through LGT adapter
        visual_features = self.encode_video(visual, padding_mask, lengths)  # [B, T, 512]

        # Encode text prompts
        text_features_ori = self.encode_textprompt(text)  # [C, 512] = [4, 512]


        # Visual prompt fusion
        text_features = text_features_ori.unsqueeze(0).expand(visual_features.shape[0], -1, -1)  # [B, C, 512]

        # Attention-weighted visual features
        visual_attn = visual_features + self.mlp2(visual_features)  # [B, T, 512]
        visual_attn_norm = l2_normalize(visual_attn, dim=-1)

        # Fuse visual and text
        # Cross-attention: text Query retrieves visual Key/Value according to semantic needs, retaining frame-level temporal weights
        text_features = self.cross_fusion(text_features, visual_attn_norm)  # [B, C, 512]
        text_features = text_features + self.mlp1(text_features)  # [B, C, 512]

        # Compute similarity
        visual_features_norm = l2_normalize(visual_features, dim=-1)  # [B, T, 512]
        text_features_norm = l2_normalize(text_features, dim=-1)  # [B, C, 512]


        # Cache modal features for text-aware contrastive loss
        self.visual_features = visual_features_norm
        self.text_features = text_features_norm


        # [B, T, 512] @ [B, 512, C] = [B, T, C] = [128, 512, 4]
        logits = visual_features_norm @ text_features_norm.permute(0, 2, 1) / 0.07

        # Split logits by number of positive and negative prompts
        num_pos = len(self.positive_prompts)
        num_neg = len(self.negative_prompts)

        logits_pos_all = logits[:, :, :num_pos]                      # [B, T, num_pos]
        logits_neg_all = logits[:, :, num_pos:num_pos + num_neg]     # [B, T, num_neg]

        # Take max for each group (supports >2 prompts per type)
        logits_pos_max = logits_pos_all.max(dim=2, keepdim=True)[0]  # [B, T, 1]
        logits_neg_max = logits_neg_all.max(dim=2, keepdim=True)[0]  # [B, T, 1]

        # Anomaly score A = S_neg - S_pos
        anomaly_scores = (logits_neg_max - logits_pos_max).squeeze(-1)  # [B, T]

        # Store for loss computation during training
        self.logits_neg_max = logits_neg_max
        self.anomaly_scores = anomaly_scores

        # Return 3 values for test compatibility
        return text_features_ori, logits_pos_max, logits
