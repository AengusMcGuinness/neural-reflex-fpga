"""
model.py

Seed-conditioned Residual MLP for SO-101 Inverse Kinematics.

Architecture (per specification -- identical hidden stack to the UR5e v3
model in ES_259_final_project/ik_v3/model.py; only the input/output widths
change because the SO-101 has 5 kinematic joints):
  Input:  14D = 3D position + 6D rotation + 5D seed joints
  Hidden: 256 -> 512 -> 512 -> 256  (Mish + BatchNorm + residual skips)
  Output: 10D = [sin(q1), cos(q1), ..., sin(q5), cos(q5)]

Key design choices:
  - Mish activation (smooth, no dead neurons)
  - Batch normalization for stable training
  - Additive residual connections between matching-dimension layers
  - No output activation (raw sin/cos prediction)
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear -> BatchNorm -> Mish with optional residual addition."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.Mish()
        self.use_residual = (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.linear(x)))
        if self.use_residual:
            out = out + x
        return out


class SeedConditionedIKModel(nn.Module):
    """Residual MLP for seed-conditioned IK prediction.

    Input:  (N, 14) -- [pos_3d, rot_6d, seed_joints_5d]
    Output: (N, 10) -- [sin(q1), cos(q1), ..., sin(q5), cos(q5)]

    Architecture:
        14 -> 256 -> 512 -> 512 -> 256 -> 10
               |                    ^
               +-- residual skip ---+  (256 -> 256)
                        |      ^
                        +------+  (512 -> 512)
    """

    INPUT_DIM = 14
    OUTPUT_DIM = 10

    def __init__(self):
        super().__init__()

        # Block 1: 14 -> 256 (no residual, dimension change)
        self.block1 = ResidualBlock(self.INPUT_DIM, 256)

        # Block 2: 256 -> 512 (no residual, dimension change)
        self.block2 = ResidualBlock(256, 512)

        # Block 3: 512 -> 512 (residual: adds block2 output)
        self.block3 = ResidualBlock(512, 512)

        # Block 4: 512 -> 256 (no residual, dimension change)
        self.block4_linear = nn.Linear(512, 256)
        self.block4_bn = nn.BatchNorm1d(256)
        self.block4_act = nn.Mish()

        # Output: 256 -> 10  (no activation)
        self.output_layer = nn.Linear(256, self.OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 14) -- concatenation of [position, rotation_6d, seed_joints]

        Returns
        -------
        sincos : (N, 10) -- predicted [sin(q1), cos(q1), ..., sin(q5), cos(q5)]
        """
        # 14 -> 256
        h1 = self.block1(x)          # (N, 256), save for skip

        # 256 -> 512
        h2 = self.block2(h1)         # (N, 512)

        # 512 -> 512 + residual from h2
        h3 = self.block3(h2)         # (N, 512) + h2 via internal residual

        # 512 -> 256 + residual skip from h1
        h4 = self.block4_act(self.block4_bn(self.block4_linear(h3)))
        h4 = h4 + h1                 # additive skip: 256 -> 256

        # 256 -> 10
        out = self.output_layer(h4)
        return out


def build_model() -> SeedConditionedIKModel:
    """Construct the model and print parameter count."""
    model = SeedConditionedIKModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SeedConditionedIKModel: {n_params:,} trainable parameters")
    print(f"  Input:  {model.INPUT_DIM}D "
          f"(3D pos + 6D rot + 5D seed joints)")
    print(f"  Output: {model.OUTPUT_DIM}D "
          f"(sin/cos pairs for 5 joints)")
    print(f"  Architecture: 14 -> 256 -> 512 -> 512 -> 256 -> 10")
    print(f"  Activations: Mish | Norm: BatchNorm | Residual skips")
    return model
