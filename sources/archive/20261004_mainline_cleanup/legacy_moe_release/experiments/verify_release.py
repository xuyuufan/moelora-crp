"""Verify every immutable model, dataset, and result shipped in the release."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scrp.adaptation_data import manifest_sha256


def verify(checksum_file: str | Path) -> tuple[Path, ...]:
    """Validate all listed files and return their resolved paths."""

    checksum_path = Path(checksum_file).resolve()
    root = checksum_path.parent
    verified: list[Path] = []
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="ascii").splitlines(), start=1
    ):
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"invalid checksum line {line_number}") from error
        path = (root / relative).resolve()
        # A checksum file must not be able to escape its release directory.
        if root != path and root not in path.parents:
            raise ValueError(f"checksum path escapes release root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = manifest_sha256(path)
        if actual != expected.lower():
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual}")
        verified.append(path)
    if not verified:
        raise ValueError("checksum file contains no artifacts")
    return tuple(verified)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checksum_file", nargs="?", default="SHA256SUMS")
    args = parser.parse_args(argv)
    verified = verify(args.checksum_file)
    print(f"verified {len(verified)} release artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
