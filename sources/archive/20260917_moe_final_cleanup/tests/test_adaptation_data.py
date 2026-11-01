from __future__ import annotations

import pytest

from scrp.adaptation_data import (
    HIGH_DENSITY_DOMAIN,
    LARGE_SCALE_DOMAIN,
    load_and_validate_target_domain,
    write_target_domain_splits,
)


@pytest.mark.parametrize("domain", [LARGE_SCALE_DOMAIN.name, HIGH_DENSITY_DOMAIN.name])
def test_target_domain_splits_are_disjoint_and_meet_declared_ranges(tmp_path, domain: str) -> None:
    result = write_target_domain_splits(
        domain,
        train_output=tmp_path / "train.json",
        validation_output=tmp_path / "validation.json",
        sealed_test_output=tmp_path / "test_sealed.json",
        master_seed=41,
        train_count=12,
        validation_count=5,
        sealed_test_count=7,
    )
    assert result["adapt_train"]["episodes"] == 12
    assert result["validation"]["episodes"] == 5
    assert result["test_sealed"]["episodes"] == 7
    train = load_and_validate_target_domain(
        tmp_path / "train.json", domain=domain, allowed_splits=("adapt_train",)
    )
    validation = load_and_validate_target_domain(
        tmp_path / "validation.json", domain=domain, allowed_splits=("validation",)
    )
    test = load_and_validate_target_domain(
        tmp_path / "test_sealed.json", domain=domain, allowed_splits=("test_sealed",)
    )
    ids = [episode.canonical_id for episode in (*train, *validation, *test)]
    assert len(ids) == len(set(ids))


def test_sealed_test_manifest_cannot_be_loaded_as_adaptation_data(tmp_path) -> None:
    write_target_domain_splits(
        LARGE_SCALE_DOMAIN.name,
        train_output=tmp_path / "train.json",
        validation_output=tmp_path / "validation.json",
        sealed_test_output=tmp_path / "test.json",
        master_seed=4,
        train_count=2,
        validation_count=2,
        sealed_test_count=2,
    )
    with pytest.raises(ValueError, match="split is not permitted"):
        load_and_validate_target_domain(
            tmp_path / "test.json",
            domain=LARGE_SCALE_DOMAIN.name,
            allowed_splits=("adapt_train",),
        )


def test_fixed_target_manifests_are_not_overwritten(tmp_path) -> None:
    write_target_domain_splits(
        HIGH_DENSITY_DOMAIN.name,
        train_output=tmp_path / "train.json",
        validation_output=tmp_path / "validation.json",
        sealed_test_output=tmp_path / "test.json",
        master_seed=10,
        train_count=2,
        validation_count=2,
        sealed_test_count=2,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_target_domain_splits(
            HIGH_DENSITY_DOMAIN.name,
            train_output=tmp_path / "train.json",
            validation_output=tmp_path / "validation.json",
            sealed_test_output=tmp_path / "test.json",
            master_seed=11,
            train_count=2,
            validation_count=2,
            sealed_test_count=2,
        )
