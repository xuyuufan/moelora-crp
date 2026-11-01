"""Strict loading of the immutable V1 H=20 base policy."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .adaptation_data import manifest_sha256
from .scale_diverse_policy import SCALE_DIVERSE_POLICY_V1, ScaleDiverseRelocationPolicy


def load_h20_base_policy(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_lookahead_horizon: int = 20,
) -> tuple[ScaleDiverseRelocationPolicy, dict[str, object], str]:
    """Load the immutable V1 H=20 base policy and return its file hash."""

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("base checkpoint must be a dictionary")
    if payload.get("policy_architecture") != SCALE_DIVERSE_POLICY_V1:
        raise ValueError("MoE-LoRA requires a V1 relocation-only base checkpoint")
    config = payload.get("online_crp_training_config")
    if not isinstance(config, Mapping):
        raise ValueError("base checkpoint has no online CRP training configuration")
    horizon = int(config.get("lookahead_horizon", -1))
    if horizon != expected_lookahead_horizon:
        raise ValueError(
            f"base checkpoint lookahead horizon is {horizon}, expected {expected_lookahead_horizon}"
        )
    policy = ScaleDiverseRelocationPolicy(
        lookahead_horizon=horizon,
        embed_dim=int(config["embed_dim"]),
        num_encoder_layers=int(config["num_encoder_layers"]),
        num_heads=int(config["num_heads"]),
        ffn_dim=int(config["ffn_dim"]),
        clip_constant=float(config["clip_constant"]),
        validate_inputs=bool(config.get("runtime_validation", False)),
    ).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("base checkpoint has no model state dictionary")
    policy.load_state_dict(state, strict=True)
    return policy, dict(payload), manifest_sha256(path)


__all__ = ["load_h20_base_policy"]
