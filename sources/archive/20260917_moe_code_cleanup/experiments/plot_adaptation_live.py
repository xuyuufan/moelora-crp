"""Render and optionally watch live LoRA/MoE-LoRA adaptation metrics."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def load_adaptation_epochs(path: str | Path) -> list[dict[str, Any]]:
    """Read complete epoch records while tolerating a partial final JSONL line."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
        if isinstance(record, dict) and record.get("record_type") == "epoch":
            records.append(record)
    records.sort(key=lambda record: int(record["epoch"]))
    if not records:
        raise ValueError(f"no completed epoch records found in {source}")
    return records


def _optional_series(
    records: Sequence[Mapping[str, Any]], key: str
) -> np.ndarray:
    values = []
    for record in records:
        value = record.get(key)
        values.append(float(value) if value is not None else float("nan"))
    return np.asarray(values, dtype=float)


def _draw_curve(records: Sequence[Mapping[str, Any]]) -> plt.Figure:
    epochs = np.asarray([int(record["epoch"]) for record in records], dtype=int)
    validation = _optional_series(
        records, "validation_mean_relocations_per_container"
    )
    sampled_relocations = _optional_series(
        records, "epoch_mean_sampled_relocations"
    )
    losses = _optional_series(records, "epoch_mean_normalized_loss")
    entropy = _optional_series(records, "epoch_router_mean_entropy")
    expert_count = max(
        (
            len(weights)
            for record in records
            if isinstance((weights := record.get("epoch_router_mean_weights")), list)
        ),
        default=0,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    validation_axis, relocation_axis, loss_axis, routing_axis = axes.flat

    validation_axis.plot(
        epochs,
        validation,
        color="#DC2626",
        linewidth=2.2,
        marker="o",
        label="Validation mean R/N",
    )
    finite_validation = np.flatnonzero(np.isfinite(validation))
    if finite_validation.size:
        best = int(finite_validation[np.argmin(validation[finite_validation])])
        validation_axis.scatter(
            epochs[best], validation[best], s=70, color="#15803D", zorder=5
        )
        validation_axis.annotate(
            f"best {validation[best]:.4f}",
            (epochs[best], validation[best]),
            xytext=(7, 8),
            textcoords="offset points",
            color="#166534",
            fontsize=9,
        )
    validation_axis.set_title("Validation relocations per container")
    validation_axis.set_ylabel("R/N (lower is better)")
    validation_axis.legend(loc="best")

    relocation_axis.plot(
        epochs,
        sampled_relocations,
        color="#2563EB",
        linewidth=2.0,
        marker="o",
    )
    relocation_axis.set_title("Sampled training relocations")
    relocation_axis.set_ylabel("Mean relocations")

    loss_axis.plot(
        epochs,
        losses,
        color="#7C3AED",
        linewidth=2.0,
        marker="o",
        label="Total replay loss",
    )
    loss_axis.set_title("Training loss")
    loss_axis.set_ylabel("Normalized loss")
    loss_axis.legend(loc="best")

    if expert_count:
        for expert in range(expert_count):
            weights = np.asarray(
                [
                    float(record["epoch_router_mean_weights"][expert])
                    if isinstance(record.get("epoch_router_mean_weights"), list)
                    else float("nan")
                    for record in records
                ]
            )
            routing_axis.plot(
                epochs,
                weights,
                linewidth=2.0,
                marker="o",
                label=f"Expert {expert}",
            )
    entropy_axis = routing_axis.twinx()
    entropy_axis.plot(
        epochs,
        entropy,
        color="#64748B",
        linestyle="--",
        linewidth=1.5,
        label="Router entropy",
    )
    routing_axis.set_title("Dense router utilization")
    routing_axis.set_ylabel("Mean routing weight")
    routing_axis.set_ylim(0.0, 1.0)
    entropy_axis.set_ylabel("Entropy")
    handles, labels = routing_axis.get_legend_handles_labels()
    entropy_handles, entropy_labels = entropy_axis.get_legend_handles_labels()
    if handles or entropy_handles:
        routing_axis.legend(
            handles + entropy_handles, labels + entropy_labels, loc="best"
        )

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    first_epoch, last_epoch = int(epochs[0]), int(epochs[-1])
    figure.suptitle(
        f"Live MoE-LoRA Adaptation (through epoch {last_epoch})",
        fontsize=16,
        y=1.01,
    )
    if first_epoch == last_epoch:
        for axis in axes.flat:
            axis.set_xlim(first_epoch - 0.5, last_epoch + 0.5)
    figure.tight_layout()
    return figure


def update_live_adaptation_curve(
    records: Sequence[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Atomically replace the live PNG after a successful render."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = _draw_curve(records)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        figure.savefig(
            temporary_path,
            format="png",
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _process_is_running(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
            handle, ctypes.byref(exit_code)
        ):
            return False
        return exit_code.value == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]


def watch(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    training_pid: int,
    poll_seconds: float = 10.0,
) -> None:
    """Refresh when metrics change and stop after run completion or process exit."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    metrics = Path(metrics_path)
    last_signature: tuple[int, int] | None = None
    while True:
        if metrics.exists():
            stat = metrics.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                try:
                    records = load_adaptation_epochs(metrics)
                except ValueError as error:
                    if "no completed epoch records found" not in str(error):
                        raise
                else:
                    update_live_adaptation_curve(records, output_path)
                    last_signature = signature
            text_tail = metrics.read_text(encoding="utf-8")[-4096:]
            if '"record_type": "run_complete"' in text_tail:
                break
        if not _process_is_running(training_pid):
            if metrics.exists():
                try:
                    records = load_adaptation_epochs(metrics)
                except ValueError as error:
                    if "no completed epoch records found" not in str(error):
                        raise
                else:
                    update_live_adaptation_curve(records, output_path)
            break
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.watch_pid is None:
        update_live_adaptation_curve(
            load_adaptation_epochs(args.metrics), args.output
        )
    else:
        watch(
            args.metrics,
            args.output,
            training_pid=args.watch_pid,
            poll_seconds=args.poll_seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
