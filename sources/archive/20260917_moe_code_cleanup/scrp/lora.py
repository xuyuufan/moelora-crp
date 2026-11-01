"""Generic LoRA layers and adapters for explicitly selected linear projections.

The base policy stays structurally compatible with historical V1 checkpoints.
LoRA is installed explicitly only for an adaptation run, so old checkpoints can
still be loaded into :class:`ScaleDiverseRelocationPolicy` without adapters.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


LORA_ADAPTER_FORMAT = "mijnrl_lora_adapter_v1"


class LoRALinear(nn.Linear):
    """A linear layer with a zero-initialized low-rank additive update.

    ``weight`` and ``bias`` retain the names and shapes of ``nn.Linear``. This
    allows a historical model state dict to load after the layer has been
    replaced, while adapter weights remain separate under ``lora_A`` and
    ``lora_B``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must lie in [0, 1)")
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.lora_dropout = nn.Dropout(dropout)
        self.merged = False

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        if isinstance(layer, LoRALinear):
            raise ValueError("layer already has LoRA installed")
        replacement = cls(
            layer.in_features,
            layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=layer.bias is not None,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(layer.weight)
            if layer.bias is not None and replacement.bias is not None:
                replacement.bias.copy_(layer.bias)
        return replacement

    def delta_weight(self) -> torch.Tensor:
        """Return the unmerged weight update in the base weight's dtype."""

        return (self.lora_B @ self.lora_A) * self.scaling

    def merge_(self) -> None:
        """Fold the LoRA update into ``weight`` for no-adapter-overhead inference."""

        if not self.merged:
            with torch.no_grad():
                self.weight.add_(self.delta_weight().to(self.weight.dtype))
            self.merged = True

    def unmerge_(self) -> None:
        """Undo :meth:`merge_`; useful only for tests or resumed adapter training."""

        if self.merged:
            with torch.no_grad():
                self.weight.sub_(self.delta_weight().to(self.weight.dtype))
            self.merged = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = F.linear(inputs, self.weight, self.bias)
        if self.merged:
            return result
        update = F.linear(F.linear(self.lora_dropout(inputs), self.lora_A), self.lora_B)
        return result + update * self.scaling


def sha256_file(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _get_parent_and_name(module: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise ValueError(f"{dotted_name!r} does not resolve to a module")
    return parent, parts[-1]


def install_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Sequence[str],
) -> tuple[str, ...]:
    """Replace selected ``nn.Linear`` modules by zero-delta LoRA equivalents."""

    # Local import avoids a module cycle while preventing either adapter family
    # from silently wrapping or coexisting with the other one.
    from .moe_lora import iter_moe_lora_layers

    if list(iter_moe_lora_layers(model)):
        raise ValueError("generic LoRA cannot be mixed with MoE-LoRA")

    target_modules = tuple(target_modules)
    if not target_modules:
        raise ValueError("LoRA target module list must not be empty")
    if len(set(target_modules)) != len(target_modules):
        raise ValueError("LoRA target module list must not contain duplicates")
    installed: list[str] = []
    for name in target_modules:
        parent, attribute = _get_parent_and_name(model, name)
        original = getattr(parent, attribute)
        if isinstance(original, LoRALinear):
            raise ValueError(f"LoRA is already installed at {name}")
        if not isinstance(original, nn.Linear):
            raise TypeError(f"LoRA target {name} must be nn.Linear, got {type(original)!r}")
        setattr(
            parent,
            attribute,
            LoRALinear.from_linear(
                original, rank=rank, alpha=alpha, dropout=dropout
            ),
        )
        installed.append(name)
    # Preserve the requested target order in saved adapters; named_modules()
    # would instead return the model's structural order.
    model._lora_target_modules = tuple(installed)  # type: ignore[attr-defined]
    return tuple(installed)


def iter_lora_layers(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            yield name, layer


def configure_trainable_parameters(model: nn.Module) -> tuple[str, ...]:
    """Freeze the base and train only explicitly installed LoRA matrices."""
    layers = list(iter_lora_layers(model))
    if not layers:
        raise ValueError("install_lora() is required before configuring training")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name, layer in layers:
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
        trainable.extend((f"{name}.lora_A", f"{name}.lora_B"))
    return tuple(trainable)


def parameter_counts(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_fraction": trainable / total if total else 0.0,
    }


def lora_config(model: nn.Module) -> dict[str, object]:
    layers = list(iter_lora_layers(model))
    if not layers:
        raise ValueError("model has no installed LoRA layers")
    first = layers[0][1]
    installed = getattr(model, "_lora_target_modules", None)
    targets = tuple(installed) if installed is not None else tuple(name for name, _ in layers)
    if set(targets) != {name for name, _ in layers}:
        raise ValueError("recorded LoRA target list does not match installed layers")
    return {
        "target_modules": list(targets),
        "rank": first.rank,
        "alpha": first.alpha,
        "dropout": first.lora_dropout.p,
    }


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only low-rank parameters, cloned onto CPU for portable storage."""

    state: dict[str, torch.Tensor] = {}
    for name, layer in iter_lora_layers(model):
        if layer.merged:
            raise ValueError("unmerge LoRA before saving a trainable adapter")
        state[f"{name}.lora_A"] = layer.lora_A.detach().cpu().clone()
        state[f"{name}.lora_B"] = layer.lora_B.detach().cpu().clone()
    if not state:
        raise ValueError("model has no installed LoRA layers")
    return state


def save_lora_adapter(
    path: str | Path,
    model: nn.Module,
    *,
    base_checkpoint_sha256: str,
    metadata: Mapping[str, object],
) -> Path:
    """Save an adapter without duplicating frozen base weights."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": LORA_ADAPTER_FORMAT,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "lora_config": lora_config(model),
            "adapter_state_dict": lora_state_dict(model),
            "metadata": dict(metadata),
        },
        destination,
    )
    return destination


def load_lora_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    expected_base_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Install and load a standalone adapter, validating its base identity."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != LORA_ADAPTER_FORMAT:
        raise ValueError("not a MijnRL LoRA adapter checkpoint")
    base_hash = payload.get("base_checkpoint_sha256")
    if not isinstance(base_hash, str):
        raise ValueError("adapter has no base checkpoint hash")
    if expected_base_checkpoint_sha256 is not None and base_hash != expected_base_checkpoint_sha256:
        raise ValueError("adapter base checkpoint hash does not match the loaded base")
    config = payload.get("lora_config")
    if not isinstance(config, Mapping):
        raise ValueError("adapter has no LoRA configuration")
    targets = tuple(str(name) for name in config.get("target_modules", ()))
    if not targets:
        raise ValueError("adapter target module list is empty")
    install_lora(
        model,
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config.get("dropout", 0.0)),
        target_modules=targets,
    )
    adapter_state = payload.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("adapter has no state dictionary")
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    expected = {f"{name}.lora_{matrix}" for name in targets for matrix in ("A", "B")}
    missing_adapter = expected.intersection(missing)
    unexpected_adapter = set(unexpected)
    if missing_adapter or unexpected_adapter:
        raise ValueError(
            "adapter state mismatch: "
            f"missing={sorted(missing_adapter)}, unexpected={sorted(unexpected_adapter)}"
        )
    return dict(payload)


def merge_lora_(model: nn.Module) -> tuple[str, ...]:
    """Merge every installed adapter in place and return merged module names."""

    merged: list[str] = []
    for name, layer in iter_lora_layers(model):
        layer.merge_()
        merged.append(name)
    if not merged:
        raise ValueError("model has no installed LoRA layers")
    return tuple(merged)


__all__ = [
    "LORA_ADAPTER_FORMAT",
    "LoRALinear",
    "configure_trainable_parameters",
    "install_lora",
    "iter_lora_layers",
    "load_lora_adapter",
    "lora_config",
    "lora_state_dict",
    "merge_lora_",
    "parameter_counts",
    "save_lora_adapter",
    "sha256_file",
]
