"""FoldPath with multi-source feature fusion for furniture finetuning."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .foldpath import FoldPath, FoldPathConfig, PointNet2Encoder


class SimpleFeatureAdapter(nn.Module):
    """simple adapter"""
    
    def __init__(self, d_model: int = 384, adapter_dim: int = 128):
        super().__init__()
        self.down_proj = nn.Linear(d_model, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, d_model)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return self.layer_norm(x + residual)


class AttentionFusion(nn.Module):
    """注意力机制融合多个源模型特征"""
    
    def __init__(self, d_model: int = 384, num_sources: int = 4):
        super().__init__()
        self.num_sources = num_sources
        self.query = nn.Linear(d_model, d_model)
        self.keys = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_sources)])
        self.values = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_sources)])
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of (B, 256, d_model) from different source models
        Returns:
            fused: (B, 256, d_model)
        """
        B, N, C = features[0].shape
        
        # Query from the first feature (or average)
        q = self.query(features[0].mean(dim=1, keepdim=True))  # (B, 1, C)
        
        # Compute attention scores
        attn_scores = []
        for i in range(self.num_sources):
            k = self.keys[i](features[i].mean(dim=1, keepdim=True))  # (B, 1, C)
            score = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
            attn_scores.append(score)
        
        attn_scores = torch.cat(attn_scores, dim=-1)  # (B, 1, num_sources)
        attn_weights = self.softmax(attn_scores)
        
        # Weighted combination
        weighted_features = []
        for i in range(self.num_sources):
            v = self.values[i](features[i])  # (B, N, C)
            weight = attn_weights[:, :, i:i+1]  # (B, 1, 1)
            weighted_features.append(v * weight)
        
        fused = sum(weighted_features) / self.num_sources
        fused = self.out_proj(fused)
        return fused


class FoldPathFusion(nn.Module):
    """FoldPath with multi-source fusion for furniture finetuning."""
    
    def __init__(
        self, 
        source_model_paths: List[str],
        cfg: Optional[FoldPathConfig] = None,
        freeze_sources: bool = True,
        use_adapter: bool = True,
        adapter_dim: int = 128
    ):
        super().__init__()
        self.cfg = cfg or FoldPathConfig()
        self.source_model_paths = source_model_paths
        self.freeze_sources = freeze_sources
        self.use_adapter = use_adapter
        
        # Load source models
        self.source_models = nn.ModuleList()
        for path in source_model_paths:
            model = FoldPath(self.cfg)
            checkpoint = torch.load(path, map_location='cpu')
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            self.source_models.append(model)
        
        # Freeze source models if needed
        if freeze_sources:
            for model in self.source_models:
                for param in model.parameters():
                    param.requires_grad = False
        
        # Adapter modules (optional)
        if use_adapter:
            self.adapters = nn.ModuleList([
                SimpleFeatureAdapter(self.cfg.d_model, adapter_dim)
                for _ in range(len(source_model_paths))
            ])
        else:
            self.adapters = None
        
        # Fusion module
        self.fusion = AttentionFusion(self.cfg.d_model, len(source_model_paths))
        
        # Keep original decoder and head (these will be trained)
        self.decoder = self.source_models[0].decoder
        self.query_embed = self.source_models[0].query_embed
        self.head = self.source_models[0].head
        self.matcher = self.source_models[0].matcher
        
        # Initialize new parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize new modules"""
        for p in self.fusion.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.adapters is not None:
            for adapter in self.adapters:
                for p in adapter.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
    
    def encode_sources(self, xyz: torch.Tensor) -> List[torch.Tensor]:
        """Get features from all source models"""
        features = []
        for i, model in enumerate(self.source_models):
            with torch.set_grad_enabled(not self.freeze_sources):
                _, z = model.encoder(xyz)  # (B, 256, C)
                if self.adapters is not None:
                    z = self.adapters[i](z)
                features.append(z)
        return features
    
    def forward(self, pc: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare input
        if pc.dim() != 3:
            raise ValueError("pc must be a 3D tensor")
        if pc.shape[1] == 3:
            xyz = pc
        else:
            xyz = pc.permute(0, 2, 1).contiguous()
        
        # Get features from all source models
        source_features = self.encode_sources(xyz)
        
        # Fuse features
        fused_features = self.fusion(source_features)  # (B, 256, C)
        memory = fused_features.permute(1, 0, 2).contiguous()  # (256, B, C)
        
        # Decode paths (same as original FoldPath)
        Q = self.query_embed.weight.unsqueeze(1).repeat(1, pc.shape[0], 1)  # (num_queries, B, C)
        P = self.decoder(tgt=Q, memory=memory)  # (num_queries, B, C)
        P = P.permute(1, 0, 2).contiguous()  # (B, Q, C)
        
        # Generate paths through head
        B, Qn, C = P.shape
        P_flat = P.reshape(B * Qn, C)
        s_flat = s.unsqueeze(1).repeat(1, Qn, 1, 1).reshape(B * Qn, s.shape[1], 1)
        y_flat, f_flat = self.head(P_flat, s_flat)
        
        y_hat = y_flat.reshape(B, Qn, s.shape[1], 6)
        f_hat = f_flat.reshape(B, Qn)
        
        return y_hat, f_hat
    
    def loss(self, y_hat: torch.Tensor, f_hat: torch.Tensor, y_gt: torch.Tensor, f_gt: torch.Tensor):
        """Use original loss function"""
        # Reuse the first source model's loss function
        return self.source_models[0].loss(y_hat, f_hat, y_gt, f_gt)
    
    @torch.no_grad()
    def infer(self, pc: torch.Tensor, **kwargs):
        """Inference wrapper"""
        return self.source_models[0].infer(pc, **kwargs)
    
    def get_trainable_parameters(self):
        """Get parameters that require gradients"""
        params = []
        if not self.freeze_sources:
            params.extend(self.source_models.parameters())
        if self.adapters is not None:
            params.extend(self.adapters.parameters())
        params.extend(self.fusion.parameters())
        params.extend(self.decoder.parameters())
        params.extend(self.head.parameters())
        return params