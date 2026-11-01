"""Plot live epoch-level training and validation reward curves.

The module can be imported by the trainer to atomically refresh a PNG after
each epoch, or run as a command-line tool against an epoch-metrics JSONL file.
Legacy summary records containing ``mean_relocations`` remain supported.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

# This module is imported by a long-running, non-interactive training process.
# Select the backend before importing pyplot so plotting never requires a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def load_epoch_metrics(path: Path) -> list[dict[str, Any]]:
    """Load epoch metric records from JSONL with useful line-number errors."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"expected a JSON object in {path} at line {line_number}"
                )
            if record.get("record_type") not in {None, "epoch"}:
                continue
            records.append(record)
    if not records:
        raise ValueError(f"no epoch metric records found in {path}")
    return records


def _optional_finite_float(
    record: Mapping[str, Any], key: str, *, record_label: str
) -> float:
    value = record.get(key)
    if value is None:
        return float("nan")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{record_label} has a non-numeric {key}: {value!r}") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{record_label} has a non-finite {key}: {value!r}")
    return numeric


def _normalise_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("at least one epoch metric record is required")

    normalised: list[tuple[int, Mapping[str, Any], float, float]] = []
    seen_epochs: set[int] = set()
    for position, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {position} is not a mapping")
        if "epoch" not in record:
            raise ValueError(f"record {position} is missing epoch")
        epoch_value = record["epoch"]
        if isinstance(epoch_value, bool):
            raise ValueError(f"record {position} has an invalid epoch: {epoch_value!r}")
        try:
            epoch = int(epoch_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"record {position} has an invalid epoch: {epoch_value!r}"
            ) from exc
        try:
            epoch_as_float = float(epoch_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"record {position} has an invalid epoch: {epoch_value!r}"
            ) from exc
        if epoch < 1 or epoch_as_float != epoch:
            raise ValueError(f"record {position} has an invalid epoch: {epoch_value!r}")
        if epoch in seen_epochs:
            raise ValueError(f"duplicate epoch metric record: {epoch}")
        seen_epochs.add(epoch)

        label = f"epoch {epoch}"
        if "train_macro_reward" in record:
            train_reward = _optional_finite_float(
                record, "train_macro_reward", record_label=label
            )
        elif "mean_reward" in record:
            train_reward = _optional_finite_float(
                record, "mean_reward", record_label=label
            )
        elif "mean_relocations" in record:
            train_reward = -_optional_finite_float(
                record, "mean_relocations", record_label=label
            )
        else:
            raise ValueError(
                f"{label} is missing train_macro_reward (or legacy mean_relocations)"
            )
        if not np.isfinite(train_reward):
            raise ValueError(f"{label} has no train reward")

        validation_reward = _optional_finite_float(
            record, "validation_macro_reward", record_label=label
        )
        normalised.append((epoch, record, train_reward, validation_reward))

    normalised.sort(key=lambda item: item[0])
    sorted_records = [item[1] for item in normalised]
    epochs = np.asarray([item[0] for item in normalised], dtype=int)
    train_rewards = np.asarray([item[2] for item in normalised], dtype=float)
    validation_rewards = np.asarray([item[3] for item in normalised], dtype=float)
    return sorted_records, epochs, train_rewards, validation_rewards


def load_epoch_rewards(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load training rewards, retaining the legacy two-array public API."""

    _, epochs, train_rewards, _ = _normalise_records(load_epoch_metrics(path))
    return epochs, train_rewards


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing mean, leaving gaps until a complete finite window exists."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window < 1:
        raise ValueError("window must be positive")
    result = np.full(values.shape, np.nan, dtype=float)
    for end in range(window, len(values) + 1):
        current = values[end - window : end]
        if np.isfinite(current).all():
            result[end - 1] = float(np.mean(current))
    return result


def _best_reward_index(
    train_rewards: np.ndarray, validation_rewards: np.ndarray
) -> tuple[int, str]:
    validation_indices = np.flatnonzero(np.isfinite(validation_rewards))
    if validation_indices.size:
        relative_index = int(np.argmax(validation_rewards[validation_indices]))
        return int(validation_indices[relative_index]), "validation"
    return int(np.argmax(train_rewards)), "training"


def _stop_marker(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    reason = record.get("early_stop_reason", record.get("stop_reason"))
    stopped = bool(
        record.get("early_stop_triggered")
        or record.get("stopped_early")
        or record.get("stop")
    ) or bool(reason)
    return stopped, None if reason is None else str(reason)


def _draw_reward_curve(
    records: Sequence[Mapping[str, Any]],
) -> plt.Figure:
    sorted_records, epochs, train_rewards, validation_rewards = _normalise_records(
        records
    )
    train_average = trailing_mean(train_rewards, window=5)
    validation_average = trailing_mean(validation_rewards, window=5)
    best_index, best_source = _best_reward_index(train_rewards, validation_rewards)
    best_values = validation_rewards if best_source == "validation" else train_rewards

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    ax.plot(
        epochs,
        train_rewards,
        color="#60A5FA",
        linewidth=1.25,
        alpha=0.55,
        marker="o",
        markersize=2.5,
        label="Train macro reward",
    )
    ax.plot(
        epochs,
        train_average,
        color="#2563EB",
        linewidth=2.1,
        label="Train 5-epoch mean",
    )

    if np.isfinite(validation_rewards).any():
        ax.plot(
            epochs,
            validation_rewards,
            color="#FB923C",
            linewidth=1.45,
            alpha=0.72,
            marker="o",
            markersize=3.0,
            label="Validation macro reward",
        )
        if np.isfinite(validation_average).any():
            ax.plot(
                epochs,
                validation_average,
                color="#DC2626",
                linewidth=2.6,
                label="Validation 5-epoch mean",
            )

    best_epoch = int(epochs[best_index])
    best_reward = float(best_values[best_index])
    ax.scatter(
        best_epoch,
        best_reward,
        s=64,
        color="#15803D",
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label=f"Best {best_source}: epoch {best_epoch}",
    )
    ax.annotate(
        f"{best_reward:.2f}",
        (best_epoch, best_reward),
        xytext=(8, 10),
        textcoords="offset points",
        color="#166534",
        fontsize=9,
        fontweight="bold",
    )

    for record, epoch in zip(sorted_records, epochs, strict=True):
        stopped, reason = _stop_marker(record)
        if not stopped:
            continue
        stop_label = "Stopped early"
        if reason:
            stop_label += f": {reason}"
        ax.axvline(
            int(epoch),
            color="#7C3AED",
            linestyle="--",
            linewidth=1.35,
            alpha=0.8,
            label=stop_label,
        )

    first_epoch = int(epochs[0])
    last_epoch = int(epochs[-1])
    if first_epoch == last_epoch:
        ax.set_xlim(first_epoch - 0.5, last_epoch + 0.5)
    else:
        padding = max(0.5, (last_epoch - first_epoch) * 0.025)
        ax.set_xlim(first_epoch - padding, last_epoch + padding)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
    ax.set_title(
        f"Live Reward Curve (through Epoch {last_epoch})", fontsize=16, pad=14
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro mean episode reward (-relocations)")
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.32)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.text(
        0.01,
        0.02,
        "Higher is better",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    return fig


def update_live_reward_curve(
    records: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    """Render *records* and atomically replace *output_path* with a PNG.

    The old output remains intact if rendering or replacement fails. Exceptions
    intentionally propagate so the trainer can log them as non-fatal warnings.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = _draw_reward_curve(records)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.stem}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        figure.savefig(
            temporary_path,
            format="png",
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        type=Path,
        help="epoch metrics JSONL (new live metrics or legacy training summary)",
    )
    parser.add_argument("output", type=Path, help="output PNG path")
    args = parser.parse_args()

    update_live_reward_curve(load_epoch_metrics(args.summary), args.output)


if __name__ == "__main__":
    main()
