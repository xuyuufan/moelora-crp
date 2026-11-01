"""Run the released MoE-LoRA experiment from training through evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


FINAL_SEEDS = (20260913, 20260923, 20260933)


def _run(arguments: Sequence[str], *, project_root: Path) -> None:
    command = [sys.executable, "-B", *arguments]
    print("+", subprocess.list2cmdline(command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=True,
    )


def reproduce(
    output_root: str | Path,
    *,
    device: str,
    precision_mode: str,
    quick_check: bool,
) -> Path:
    """Train seed(s), evaluate them, and render the final figure."""

    project_root = Path(__file__).resolve().parents[1]
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = project_root / destination
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite reproduction: {destination}")
    destination.mkdir(parents=True)

    seeds = FINAL_SEEDS[:1] if quick_check else FINAL_SEEDS
    adapters: list[tuple[str, Path]] = []
    for seed in seeds:
        run_dir = destination / f"seed{seed}"
        command = [
            "-m",
            "experiments.train_moe_lora",
            "--output-dir",
            str(run_dir),
            "--seed",
            str(seed),
            "--device",
            device,
            "--precision-mode",
            precision_mode,
        ]
        if quick_check:
            command.append("--quick-check")
        _run(command, project_root=project_root)
        adapters.append((f"MoE_seed{seed}", run_dir / "best.adapter.pt"))

    evaluation_dir = destination / "sealed_test"
    command = [
        "-m",
        "experiments.evaluate_moe_lora",
        "--output-dir",
        str(evaluation_dir),
        "--device",
        device,
    ]
    for name, path in adapters:
        command.extend(("--adapter", f"{name}={path}"))
    if quick_check:
        command.append("--quick-check")
    _run(command, project_root=project_root)

    # The publication figure has a fixed three-seed layout. A quick check
    # intentionally stops after validating the single-seed evaluation path.
    if not quick_check:
        _run(
            [
                "-m",
                "figures.gen_fig_moe_sealed_test",
                "--summary",
                str(evaluation_dir / "summary.json"),
                "--output-prefix",
                str(destination / "fig_moe_sealed_test"),
            ],
            project_root=project_root,
        )

    record = {
        "quick_check": quick_check,
        "seeds": list(seeds),
        "device": device,
        "precision_mode": precision_mode,
        "evaluation": str(evaluation_dir / "summary.json"),
    }
    (destination / "reproduction.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output/reproduction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision-mode",
        choices=("fp32", "tf32", "bf16", "fp16"),
        default="fp16",
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="one seed, one update, four test instances; not a paper result",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = reproduce(
        args.output_root,
        device=args.device,
        precision_mode=args.precision_mode,
        quick_check=args.quick_check,
    )
    print(f"reproduction complete: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
