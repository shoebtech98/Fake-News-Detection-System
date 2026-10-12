"""Checks for the data boundaries that keep evaluations meaningful."""

import pytest

import src.pipeline as pipeline
from src.pipeline import load_data, split_data


def test_sample_data_and_splits_have_both_labels_without_overlap():
    frame = load_data("sample")
    train, valid, test = split_data(frame, sample=True)
    assert len(train) + len(valid) + len(test) == len(frame)
    for part in (train, valid, test):
        assert set(part["label"]) == {0, 1}
    assert set(train["combined"]).isdisjoint(valid["combined"])
    assert set(train["combined"]).isdisjoint(test["combined"])
    assert set(valid["combined"]).isdisjoint(test["combined"])


def test_missing_real_data_has_actionable_error(monkeypatch):
    monkeypatch.setattr(pipeline, "PROCESSED_DATA", pipeline.ROOT / "data/processed/__absent_test__.csv")
    with pytest.raises(FileNotFoundError, match="README.md"):
        load_data("processed")
