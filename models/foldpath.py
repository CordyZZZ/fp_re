"""FoldPath model (Rabino et al., 2025) implemented on top of the MaskPlanner codebase.

The paper proposes an end-to-end Object-Centric Motion Generation model that:
  1) encodes an input object point cloud into visual features z;
  2) decodes N path embeddings (queries) via a transformer decoder;
  3) generates each path by sampling a continuous (implicit) function of a scalar
     parameter s \in [-1, 1] through a modulated MLP head (a "neural field"-style
     decoder).

This file implements the architecture and its training losses:
  - Hungarian matching on 3D positions
  - Point loss: L2 position + cosine angular loss for orientations
  - Confidence loss: focal loss

Notes:
  - The PaintNet dataset encodes orientations as 3D unit vectors (2-DoF). We keep
    this representation.
  - The paper evaluates multiple activation choices (ReLU/SIREN/Finer). We include
    ReLU and SIREN as faithful options. "Finer" is provided as a runnable
    approximation (learned per-layer frequency); see TODO in VariablePeriodic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hungarianMatcher import HungarianMatcher
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


@dataclass
class FoldPathConfig:
    # Core sizes (paper default: C=384, N=40, L=4, H=512)
    num_queries: int = 40
    d_model: int = 384
    head_hidden: int = 512
    head_layers: int = 4
    tf_layers: int = 4
    tf_heads: int = 4

    # Sampling
    T_train: int = 64
    T_test: int = 384

    # Loss
    focal_gamma: float = 2.0

    # Activation: "relu" | "siren" | "finer"
    activation: str = "relu"


class Sine(nn.Module):
    """SIREN-style sine activation."""

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class VariablePeriodic(nn.Module):
    """Runnable approximation of FINER-style activation.

    The FINER paper uses variable-periodic activations to tune spectral bias.
    Implementations vary; to keep this repository self-contained and runnable,
    we use a learnable scalar frequency per layer.

    TODO: If you want an exact FINER reproduction, replace this module with the
          official formulation from the FINER paper / code.
    """

    def __init__(self, init_w: float = 1.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


def cosine_angular_loss(v_gt: torch.Tensor, v_pr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - cosine similarity between two orientation vectors."""
    v_gt_n = v_gt / (v_gt.norm(dim=-1, keepdim=True) + eps)
    v_pr_n = v_pr / (v_pr.norm(dim=-1, keepdim=True) + eps)
    return 1.0 - (v_gt_n * v_pr_n).sum(dim=-1)


class PointNet2Encoder(nn.Module):
    """PointNet++ encoder producing per-point visual tokens.

    The FoldPath paper describes a PointNet++ backbone with downsampling and
    feature propagation, outputting z \in R^{256 x C}. We implement a compact
    variant that is compatible with the MaskPlanner codebase.

    Output:
        xyz256: (B, 3, 256)
        z:      (B, 256, C)
    """

    def __init__(self, d_model: int = 384, in_channel: int = 3):
        super().__init__()

        # Downsample 5120 -> 1024 -> 256
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32,
                                          in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=64,
                                          in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        # Feature propagation back to 256 points (we omit the last FP to full resolution).
        self.fp3 = PointNetFeaturePropagation(in_channel=1024 + 256, mlp=[512, d_model])

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # xyz: (B, 3, N)
        l1_xyz, l1_points = self.sa1(xyz, points=None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # l3_points: (B, 1024, 1) -> propagate to 256 points
        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # (B, d_model, 256) -> (B, 256, d_model)
        z = l2_points_fp.permute(0, 2, 1).contiguous()
        return l2_xyz, z


class ModulatedMLPHead(nn.Module):
    """Per-path neural-field head: scalar s -> 6D pose, modulated by path prototype P_j."""

    def __init__(self, d_model: int, hidden: int, layers: int, activation: str = "relu"):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.layers = layers

        # Linear stack A_l
        self.A0 = nn.Linear(1, hidden)
        self.As = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 1)])
        self.A_out = nn.Linear(hidden, 6)

        # Modulation MLPs (Eq. 6)
        self.w0 = nn.Linear(d_model, hidden)
        self.ws = nn.ModuleList([nn.Linear(hidden + d_model, hidden) for _ in range(layers - 1)])

        # Activation
        activation = activation.lower()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "siren":
            self.act = Sine(w0=1.0)
        elif activation == "finer":
            self.act = VariablePeriodic(init_w=1.0)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Confidence head (2-layer FFN + sigmoid)
        self.conf = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, Pj: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
        Pj: (B, d_model)
        s:  (B, T, 1)

        Returns:
            y: (B, T, 6)
            f: (B, 1)
        """
        B, T, _ = s.shape
        # Compute modulation vectors h_l per Eq. 6
        h0 = F.relu(self.w0(Pj))  # (B, H)
        hs = [h0]
        h_prev = h0
        for wl in self.ws:
            h_prev = F.relu(wl(torch.cat([h_prev, Pj], dim=-1)))
            hs.append(h_prev)

        # Forward the scalar through modulated blocks Eq. 5
        x = self.A0(s)  # (B, T, H)
        x = hs[0].unsqueeze(1) * self.act(x)
        for i, A in enumerate(self.As, start=1):
            x = A(x)
            x = hs[i].unsqueeze(1) * self.act(x)
        y = self.A_out(x)  # (B, T, 6)

        f = self.conf(Pj)  # (B, 1)
        return y, f


class FoldPath(nn.Module):
    def __init__(self, cfg: Optional[FoldPathConfig] = None):
        super().__init__()
        self.cfg = cfg or FoldPathConfig()

        self.encoder = PointNet2Encoder(d_model=self.cfg.d_model, in_channel=3)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.tf_heads,
            dim_feedforward=self.cfg.d_model * 4,
            batch_first=False,
            dropout=0.0,
            activation="relu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.cfg.tf_layers)

        self.query_embed = nn.Embedding(self.cfg.num_queries, self.cfg.d_model)

        # One shared head module applied per query (vectorized over queries)
        self.head = ModulatedMLPHead(
            d_model=self.cfg.d_model,
            hidden=self.cfg.head_hidden,
            layers=self.cfg.head_layers,
            activation=self.cfg.activation,
        )

        # self.matcher = HungarianMatcher(cost_class=0.0, cost_bbox=1.0, cost_giou=0.0)
        self.matcher = HungarianMatcher()

    @torch.no_grad()
    def sample_s(self, B: int, T: int, mode: str = "train", device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample scalar positions s in [-1, 1]."""
        device = device or torch.device("cpu")
        if mode == "train":
            return (torch.rand(B, T, 1, device=device) * 2.0) - 1.0
        # test: equispaced
        t = torch.linspace(-1.0, 1.0, T, device=device).view(1, T, 1)
        return t.repeat(B, 1, 1)

    def forward(self, pc: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            pc: (B, N, 3) or (B, 3, N)
            s:  (B, T, 1)

        Returns:
            y_hat: (B, Q, T, 6)
            f_hat: (B, Q)
        """
        if pc.dim() != 3:
            raise ValueError("pc must be a 3D tensor")
        if pc.shape[1] == 3:
            xyz = pc
        else:
            xyz = pc.permute(0, 2, 1).contiguous()

        _, z = self.encoder(xyz)  # z: (B, 256, C)
        memory = z.permute(1, 0, 2).contiguous()  # (256, B, C)

        Q = self.query_embed.weight.unsqueeze(1).repeat(1, pc.shape[0], 1)  # (num_queries, B, C)
        P = self.decoder(tgt=Q, memory=memory)  # (num_queries, B, C)
        P = P.permute(1, 0, 2).contiguous()  # (B, Q, C)

        B, Qn, C = P.shape
        # Vectorize head over queries by reshaping batch dimension
        P_flat = P.reshape(B * Qn, C)
        s_flat = s.unsqueeze(1).repeat(1, Qn, 1, 1).reshape(B * Qn, s.shape[1], 1)
        y_flat, f_flat = self.head(P_flat, s_flat)
        y_hat = y_flat.reshape(B, Qn, s.shape[1], 6)
        f_hat = f_flat.reshape(B, Qn)
        return y_hat, f_hat


    @torch.no_grad()
    def infer(self, pc: torch.Tensor, *, T: int = 384, max_paths: int = 40, conf_thresh: float = 0.35):
        """
        Convenience inference wrapper returning point-level trajectories, compatible with the project's
        existing visualization / offline tooling.

        Returns a list (len=B) of dicts:
        - traj_pred: (P, 6) float32
        - stroke_ids_pred: (P,) int64  (0..K-1)
        - conf_pred: (K,) float32
        """
        device = pc.device
        B = pc.shape[0]
        s = torch.linspace(-1.0, 1.0, T, device=device).view(1, T, 1).repeat(B, 1, 1)
        y_hat, f_hat = self.forward(pc, s)  # (B,Q,T,6), (B,Q)
        conf = torch.sigmoid(f_hat)         # (B,Q)

        outs = []
        for b in range(B):
            cb = conf[b]
            keep = torch.nonzero(cb >= float(conf_thresh), as_tuple=False).flatten()
            if keep.numel() == 0:
                # keep at least one path to avoid empty predictions
                keep = torch.topk(cb, k=1).indices
            # cap by max_paths
            k = min(int(max_paths), int(keep.numel()))
            # reorder by confidence desc
            keep = keep[torch.argsort(cb[keep], descending=True)[:k]]

            traj_list = []
            sid_list = []
            conf_list = []
            for new_id, q in enumerate(keep.tolist()):
                pts = y_hat[b, q]  # (T,6)
                traj_list.append(pts)
                sid_list.append(torch.full((T,), new_id, device=device, dtype=torch.long))
                conf_list.append(cb[q])

            traj_pred = torch.cat(traj_list, dim=0).detach().cpu().numpy().astype('float32')
            stroke_ids_pred = torch.cat(sid_list, dim=0).detach().cpu().numpy().astype('int64')
            conf_pred = torch.stack(conf_list, dim=0).detach().cpu().numpy().astype('float32')

            outs.append({
                'traj_pred': traj_pred,
                'stroke_ids_pred': stroke_ids_pred,
                'conf_pred': conf_pred,
            })
        return outs
        
    def loss(self, y_hat: torch.Tensor, f_hat: torch.Tensor, y_gt: torch.Tensor, f_gt: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute FoldPath losses.

        Shapes:
            y_hat: (B, Q, T, 6)
            f_hat: (B, Q)
            y_gt:  (B, Q, T, 6)  (already padded to Q)
            f_gt:  (B, Q)        (1 for real paths, 0 for padding)
        """
        B, Q, T, _ = y_hat.shape

        # Hungarian matching uses 3D positions only (Eq. 1-2). We treat each path as a "box" with
        # coordinates = mean position across T points.
        # targets is a list of dicts, as expected by the existing HungarianMatcher.

        out_pos = y_hat[..., :3].mean(dim=2)  # (B, Q, 3)
        tgt_pos = y_gt[..., :3].mean(dim=2)   # (B, Q, 3)

        # outputs = {"pred_boxes": out_pos, "pred_logits": f_hat}
        # targets = []
        # for b in range(B):
        #     # Keep only real paths for matching
        #     keep = f_gt[b] > 0.5
        #     targets.append({"boxes": tgt_pos[b, keep], "labels": torch.zeros(int(keep.sum()), device=y_gt.device, dtype=torch.long)})

        # indices = self.matcher(outputs, targets)

        targets_list = []
        for b in range(B):
            # Keep only real paths for matching
            keep = f_gt[b] > 0.5
            if keep.sum() > 0:
                target_positions = tgt_pos[b, keep]  # [num_real_paths, 3]
                targets_list.append(target_positions)
            else:
                targets_list.append(torch.empty((0, 3), device=y_gt.device))

        indices = self.matcher(out_pos, targets_list)

        # Reorder gt according to matching
        # Build an index tensor map of size (B, Q) where unmatched are left as identity
        perm = torch.arange(Q, device=y_gt.device).unsqueeze(0).repeat(B, 1)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            # src_idx: indices in predictions; tgt_idx: indices in the compacted target list
            # We need mapping from src_idx -> original target index among keep positions.
            keep = (f_gt[b] > 0.5).nonzero(as_tuple=False).squeeze(1)
            perm[b, src_idx] = keep[tgt_idx]

        y_gt_m = torch.gather(y_gt, dim=1, index=perm.view(B, Q, 1, 1).repeat(1, 1, T, 6))
        f_gt_m = torch.gather(f_gt, dim=1, index=perm)

        # Point loss Eq. 3
        mask = (f_gt_m > 0.5).float().view(B, Q, 1)
        pos_loss = ((y_gt_m[..., :3] - y_hat[..., :3]).pow(2).sum(dim=-1)).mean(dim=-1)  # (B,Q)
        ang_loss = cosine_angular_loss(y_gt_m[..., 3:6], y_hat[..., 3:6]).mean(dim=-1)   # (B,Q)
        L_points = ((pos_loss + ang_loss) * mask.squeeze(-1)).sum() / (mask.sum() + 1e-6)

        # Confidence focal loss Eq. 4
        # f_hat already in [0,1] due to sigmoid in head; clamp for stability.
        p = f_hat.clamp(1e-6, 1.0 - 1e-6)
        y = f_gt_m
        # focal loss for binary classification
        L_conf = -((1.0 - p) ** self.cfg.focal_gamma) * (y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
        L_conf = L_conf.mean()

        L = L_points + L_conf
        logs = {
            "loss": float(L.detach().cpu()),
            "loss_points": float(L_points.detach().cpu()),
            "loss_conf": float(L_conf.detach().cpu()),
        }
        return L, logs