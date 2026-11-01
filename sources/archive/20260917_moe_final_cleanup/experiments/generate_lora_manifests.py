"""Create fixed, disjoint target-domain manifests for H=20 LoRA adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from scrp.adaptation_data import TARGET_DOMAINS, write_manifest_metadata, write_target_domain_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=tuple(TARGET_DOMAINS), required=True)
    parser.add_argument(
        "--output-dir",
        help="new directory for fixed manifests; default is data/online_crp/lora_domains/<domain>",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=1_024)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--sealed-test-count", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("data/online_crp/lora_domains") / args.domain
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = write_target_domain_splits(
        args.domain,
        train_output=output_dir / "adapt_train.json",
        validation_output=output_dir / "validation.json",
        sealed_test_output=output_dir / "test_sealed.json",
        master_seed=args.seed,
        train_count=args.train_count,
        validation_count=args.validation_count,
        sealed_test_count=args.sealed_test_count,
    )
    metadata = {
        "protocol": "mijnrl_lora_target_domain_v1",
        "domain": args.domain,
        "master_seed": args.seed,
        "splits": result,
        "test_policy": "sealed; never pass test_sealed.json to adaptation or selection",
    }
    metadata_path = write_manifest_metadata(output_dir / "adapt_train.json", metadata)
    print(json.dumps({"metadata": str(metadata_path), **metadata}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
