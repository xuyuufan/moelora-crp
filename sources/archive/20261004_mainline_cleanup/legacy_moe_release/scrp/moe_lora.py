"""Dense input-routed mixtures of low-rank updates for one policy projection."""

from __future__ import annotations

import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F

MOE_LORA_TARGET_MODULE = "stack_projection.2"
MOE_LORA_ADAPTER_FORMAT = "mijnrl_moe_lora_adapter_v1"


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class LoRAExpert(nn.Module):
    """One bias-free low-rank update in a MoE-LoRA layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rank = _positive_integer(rank, "rank")
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(inputs, self.lora_A), self.lora_B)


@dataclass
class MoELoRARoutingCapture:
    """Ephemeral, graph-bearing routing data scoped to one replay chunk."""

    probabilities: list[torch.Tensor] = field(default_factory=list)
    expert_delta_norm_sums: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.probabilities.clear()
        self.expert_delta_norm_sums.clear()


class MoELoRALinear(nn.Linear):
    """A frozen linear base plus dense, input-routed LoRA experts."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_experts: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.num_experts = _positive_integer(num_experts, "num_experts")
        self.rank = _positive_integer(rank, "rank")
        self.alpha = _positive_finite(alpha, "alpha")
        if float(dropout) != 0.0:
            raise ValueError("MoE-LoRA dropout must be zero")
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.dropout = 0.0
        self.scaling = self.alpha / self.rank
        self.experts = nn.ModuleList(
            LoRAExpert(
                in_features,
                out_features,
                rank=self.rank,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.num_experts)
        )
        # Routing is deliberately FP32 even when expert/base computation is
        # executed under mixed precision.
        self.router = nn.Linear(
            in_features,
            self.num_experts,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self._routing_capture: MoELoRARoutingCapture | None = None

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        num_experts: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "MoELoRALinear":
        if isinstance(layer, MoELoRALinear):
            raise ValueError("target layer already has an adapter installed")
        replacement = cls(
            layer.in_features,
            layer.out_features,
            num_experts=num_experts,
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
        replacement.train(layer.training)
        return replacement

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            routing = torch.softmax(
                F.linear(
                    inputs.float(),
                    self.router.weight.float(),
                    self.router.bias.float(),
                ),
                dim=-1,
            )

        deltas = torch.stack(
            [expert(inputs) * self.scaling for expert in self.experts], dim=-2
        )
        mixed = (deltas.float() * routing.unsqueeze(-1)).sum(dim=-2)
        capture = self._routing_capture
        if capture is not None:
            capture.probabilities.append(routing)
            capture.expert_delta_norm_sums.append(
                deltas.float().norm(dim=-1).sum(dim=tuple(range(deltas.ndim - 2)))
            )
        return base + mixed.to(base.dtype)


def _target_parent(model: nn.Module) -> tuple[nn.Module, str]:
    parent: nn.Module = model
    parts = MOE_LORA_TARGET_MODULE.split(".")
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if not isinstance(child, nn.Module):
            raise ValueError(
                f"{MOE_LORA_TARGET_MODULE!r} does not resolve to a module"
            )
        parent = child
    return parent, parts[-1]


def iter_moe_lora_layers(
    model: nn.Module,
) -> Iterator[tuple[str, MoELoRALinear]]:
    """Yield installed MoE-LoRA layers with their fully qualified names."""

    for name, layer in model.named_modules():
        if isinstance(layer, MoELoRALinear):
            yield name, layer


def install_moe_lora(
    model: nn.Module,
    *,
    num_experts: int = 2,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> MoELoRALinear:
    """Replace the fixed fusion projection by a zero-delta MoE-LoRA layer."""

    _positive_integer(num_experts, "num_experts")
    _positive_integer(rank, "rank")
    _positive_finite(alpha, "alpha")
    if float(dropout) != 0.0:
        raise ValueError("MoE-LoRA dropout must be zero")
    if list(iter_moe_lora_layers(model)):
        raise ValueError("MoE-LoRA is already installed")
    parent, attribute = _target_parent(model)
    original = getattr(parent, attribute, None)
    if not isinstance(original, nn.Linear):
        raise TypeError(
            f"MoE-LoRA target must be nn.Linear, got {type(original)!r}"
        )
    replacement = MoELoRALinear.from_linear(
        original,
        num_experts=num_experts,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    setattr(parent, attribute, replacement)
    model._moe_lora_target_module = MOE_LORA_TARGET_MODULE  # type: ignore[attr-defined]
    return replacement


def configure_moe_lora_training(model: nn.Module) -> tuple[str, ...]:
    """Freeze the entire policy and enable only experts and router."""

    layers = list(iter_moe_lora_layers(model))
    if len(layers) != 1 or layers[0][0] != MOE_LORA_TARGET_MODULE:
        raise ValueError("install_moe_lora() is required before configuring training")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    name, layer = layers[0]
    trainable: list[str] = []
    for index, expert in enumerate(layer.experts):
        expert.lora_A.requires_grad_(True)
        expert.lora_B.requires_grad_(True)
        trainable.extend(
            (
                f"{name}.experts.{index}.lora_A",
                f"{name}.experts.{index}.lora_B",
            )
        )
    layer.router.weight.requires_grad_(True)
    layer.router.bias.requires_grad_(True)
    trainable.extend((f"{name}.router.weight", f"{name}.router.bias"))
    return tuple(trainable)


def parameter_counts(model: nn.Module) -> dict[str, int | float]:
    """Return total and trainable parameter counts for run provenance."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_fraction": trainable / total if total else 0.0,
    }


def _installed_layer(model: nn.Module) -> MoELoRALinear:
    layers = list(iter_moe_lora_layers(model))
    if len(layers) != 1 or layers[0][0] != MOE_LORA_TARGET_MODULE:
        raise ValueError("model must have exactly one MoE-LoRA target layer")
    return layers[0][1]


def moe_lora_config(model: nn.Module) -> dict[str, object]:
    """Serialize the structural contract required to reload an adapter."""

    layer = _installed_layer(model)
    return {
        "target_module": MOE_LORA_TARGET_MODULE,
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "bias": layer.bias is not None,
        "num_experts": layer.num_experts,
        "rank": layer.rank,
        "alpha": layer.alpha,
        "dropout": layer.dropout,
        "routing": "dense_per_stack_input_softmax",
        "temperature": 1.0,
    }


def moe_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy only expert and router tensors to a CPU checkpoint state."""

    layer = _installed_layer(model)
    prefix = f"{MOE_LORA_TARGET_MODULE}."
    state: dict[str, torch.Tensor] = {}
    for index, expert in enumerate(layer.experts):
        state[f"{prefix}experts.{index}.lora_A"] = (
            expert.lora_A.detach().cpu().clone()
        )
        state[f"{prefix}experts.{index}.lora_B"] = (
            expert.lora_B.detach().cpu().clone()
        )
    state[f"{prefix}router.weight"] = layer.router.weight.detach().cpu().clone()
    state[f"{prefix}router.bias"] = layer.router.bias.detach().cpu().clone()
    return state


def save_moe_lora_adapter(
    path: str | Path,
    model: nn.Module,
    *,
    base_checkpoint_sha256: str,
    metadata: Mapping[str, object],
) -> Path:
    """Atomically save a standalone adapter without duplicating base weights."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": MOE_LORA_ADAPTER_FORMAT,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "moe_lora_config": moe_lora_config(model),
        "adapter_state_dict": moe_lora_state_dict(model),
        "metadata": dict(metadata),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def load_moe_lora_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    expected_base_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Validate completely before atomically installing a saved adapter."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != MOE_LORA_ADAPTER_FORMAT:
        raise ValueError("not a MijnRL MoE-LoRA adapter checkpoint")
    base_hash = payload.get("base_checkpoint_sha256")
    if not isinstance(base_hash, str):
        raise ValueError("adapter has no base checkpoint hash")
    if (
        expected_base_checkpoint_sha256 is not None
        and base_hash != expected_base_checkpoint_sha256
    ):
        raise ValueError("adapter base checkpoint hash does not match the loaded base")
    config = payload.get("moe_lora_config")
    if not isinstance(config, Mapping):
        raise ValueError("adapter has no MoE-LoRA configuration")
    if config.get("target_module") != MOE_LORA_TARGET_MODULE:
        raise ValueError("adapter targets an unsupported module")
    if config.get("routing") != "dense_per_stack_input_softmax":
        raise ValueError("adapter has an unsupported routing rule")
    if float(config.get("temperature", float("nan"))) != 1.0:
        raise ValueError("adapter has an unsupported routing temperature")
    if float(config.get("dropout", float("nan"))) != 0.0:
        raise ValueError("adapter has nonzero dropout")
    num_experts = _positive_integer(config.get("num_experts"), "num_experts")
    rank = _positive_integer(config.get("rank"), "rank")
    alpha = _positive_finite(config.get("alpha"), "alpha")

    if list(iter_moe_lora_layers(model)):
        raise ValueError("model already has an installed adapter")
    parent, attribute = _target_parent(model)
    original = getattr(parent, attribute, None)
    if not isinstance(original, nn.Linear):
        raise TypeError("MoE-LoRA target is not a plain linear layer")
    expected_structure = {
        "in_features": original.in_features,
        "out_features": original.out_features,
        "bias": original.bias is not None,
    }
    for key, expected in expected_structure.items():
        if config.get(key) != expected:
            raise ValueError(
                f"adapter {key}={config.get(key)!r} does not match base {expected!r}"
            )

    temporary = MoELoRALinear.from_linear(
        original,
        num_experts=num_experts,
        rank=rank,
        alpha=alpha,
        dropout=0.0,
    )
    adapter_state = payload.get("adapter_state_dict")
    if not isinstance(adapter_state, Mapping):
        raise ValueError("adapter has no state dictionary")
    prefix = f"{MOE_LORA_TARGET_MODULE}."
    expected_parameters = {
        f"experts.{index}.lora_{matrix}"
        for index in range(num_experts)
        for matrix in ("A", "B")
    } | {"router.weight", "router.bias"}
    supplied: dict[str, torch.Tensor] = {}
    for full_name, value in adapter_state.items():
        if not isinstance(full_name, str) or not full_name.startswith(prefix):
            raise ValueError(f"unexpected adapter state key {full_name!r}")
        local_name = full_name[len(prefix) :]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"adapter state {full_name!r} is not a tensor")
        supplied[local_name] = value
    if set(supplied) != expected_parameters:
        raise ValueError(
            "adapter state mismatch: "
            f"missing={sorted(expected_parameters - set(supplied))}, "
            f"unexpected={sorted(set(supplied) - expected_parameters)}"
        )
    parameters = dict(temporary.named_parameters())
    with torch.no_grad():
        for name, value in supplied.items():
            target = parameters[name]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"adapter tensor {prefix + name!r} has shape {tuple(value.shape)}, "
                    f"expected {tuple(target.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))

    # No mutation of the model happens until every payload check and copy has
    # succeeded, so malformed checkpoints cannot leave a half-installed layer.
    setattr(parent, attribute, temporary)
    model._moe_lora_target_module = MOE_LORA_TARGET_MODULE  # type: ignore[attr-defined]
    return dict(payload)


@contextmanager
def capture_moe_lora_routing(
    model: nn.Module,
) -> Iterator[MoELoRARoutingCapture]:
    """Capture routing tensors only for the lexical replay/backward scope."""

    layer = _installed_layer(model)
    if layer._routing_capture is not None:
        raise RuntimeError("MoE-LoRA routing capture is already active")
    capture = MoELoRARoutingCapture()
    layer._routing_capture = capture
    try:
        yield capture
    finally:
        layer._routing_capture = None
        capture.clear()


__all__ = [
    "MOE_LORA_ADAPTER_FORMAT",
    "MOE_LORA_TARGET_MODULE",
    "LoRAExpert",
    "MoELoRALinear",
    "capture_moe_lora_routing",
    "configure_moe_lora_training",
    "install_moe_lora",
    "iter_moe_lora_layers",
    "load_moe_lora_adapter",
    "moe_lora_config",
    "moe_lora_state_dict",
    "parameter_counts",
    "save_moe_lora_adapter",
]
