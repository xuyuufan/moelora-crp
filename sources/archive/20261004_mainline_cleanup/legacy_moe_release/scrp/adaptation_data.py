"""Validation for the immutable large-scale manifests used in the paper.

The repository intentionally contains no data-generation API. Reproduction
must consume the released bytes, especially the sealed test split, rather than
silently creating a different benchmark under the same name.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from .online_crp import OnlineCRPEpisode, load_episode_manifest


TARGET_DOMAIN_PROTOCOL = "mijnrl_lora_target_domain_v1"
LARGE_SCALE_DOMAIN = "large_scale_n100_150"
TARGET_DOMAIN_SPLITS = ("adapt_train", "validation", "test_sealed")
LARGE_SCALE_SHAPES = ((20, 8), (24, 8), (20, 10), (25, 8))


def manifest_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading a checkpoint in RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_large_scale_episodes(
    episodes: Sequence[OnlineCRPEpisode], *, split: str
) -> None:
    """Reject wrong-domain, malformed, or duplicate released episodes."""

    if split not in TARGET_DOMAIN_SPLITS:
        raise ValueError(f"unsupported split {split!r}")
    if not episodes:
        raise ValueError("target-domain manifest must not be empty")
    canonical_ids = [episode.canonical_id for episode in episodes]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("target-domain manifest contains duplicate episodes")
    for episode in episodes:
        metadata = episode.metadata
        if metadata.get("protocol") != TARGET_DOMAIN_PROTOCOL:
            raise ValueError("manifest does not use the released target-domain protocol")
        if metadata.get("domain") != LARGE_SCALE_DOMAIN:
            raise ValueError("manifest is not from the final large-scale domain")
        if metadata.get("split") != split:
            raise ValueError("manifest split does not match the requested split")
        if not 100 <= episode.num_containers <= 150:
            raise ValueError("manifest container count lies outside [100, 150]")
        if not 0.60 <= episode.fill_rate <= 0.80:
            raise ValueError("manifest fill rate lies outside [0.60, 0.80]")
        if (episode.num_stacks, episode.max_tiers) not in LARGE_SCALE_SHAPES:
            raise ValueError("manifest shape is outside the released domain")


def load_large_scale_split(
    path: str | Path, split: str
) -> tuple[OnlineCRPEpisode, ...]:
    """Load and validate one fixed split from the final experiment domain."""

    episodes = load_episode_manifest(path)
    validate_large_scale_episodes(episodes, split=split)
    return episodes


__all__ = [
    "LARGE_SCALE_DOMAIN",
    "TARGET_DOMAIN_PROTOCOL",
    "TARGET_DOMAIN_SPLITS",
    "load_large_scale_split",
    "manifest_sha256",
    "validate_large_scale_episodes",
]
