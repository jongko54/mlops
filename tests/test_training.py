import json

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits

from mlops_lab.train import predict, prepare_data, train_run


def test_split_is_reproducible_disjoint_and_covers_the_dataset():
    first = prepare_data(42)
    second = prepare_data(42)
    sets = [set(first[name]["indices"].tolist()) for name in ("train", "validation", "test")]
    assert sum(map(len, sets)) == 1797
    assert len(set.union(*sets)) == 1797
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    for name in ("train", "validation", "test"):
        assert torch.equal(first[name]["x"], second[name]["x"])
        assert first[name]["x"].shape[1] == 64
        assert 0 <= first[name]["x"].min() <= first[name]["x"].max() <= 1
        assert set(first[name]["y"].tolist()) == set(range(10))
    assert not np.array_equal(first["train"]["indices"], prepare_data(7)["train"]["indices"])


def test_training_writes_reloadable_model_and_lineage(tmp_path):
    result = train_run(tmp_path / "run", epochs=30)
    assert result["validation_accuracy"] >= 0.90
    assert result["test_accuracy"] >= 0.90
    assert result["status"] == "passed"
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved == result
    assert len(saved["dataset_sha256"]) == 64
    assert len(saved["split_sha256"]) == 64
    assert saved["device"] == "cpu"
    pixels = load_digits().data[:5]
    predictions = predict(tmp_path / "run" / "model.pt", pixels)
    assert len(predictions) == 5
    assert all(0 <= p < 10 for p in predictions)
    # Completed runs must not be silently overwritten.
    with pytest.raises(FileExistsError):
        train_run(tmp_path / "run")


def test_failed_validation_does_not_promote_or_inspect_test_set(tmp_path):
    result = train_run(tmp_path / "rejected", epochs=1, min_accuracy=1.0)
    assert result["status"] == "rejected"
    assert result["test_accuracy"] is None
    assert not (tmp_path / "rejected" / "model.pt").exists()
    assert (tmp_path / "rejected" / "metrics.json").exists()


@pytest.mark.parametrize("epochs,threshold", [(0, 0.9), (-1, 0.9), (1, -0.1), (1, 1.1)])
def test_invalid_configuration_fails_before_artifacts(tmp_path, epochs, threshold):
    with pytest.raises(ValueError):
        train_run(tmp_path / "invalid", epochs=epochs, min_accuracy=threshold)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a machine without CUDA")
def test_cuda_request_never_silently_falls_back_to_cpu(tmp_path):
    with pytest.raises(RuntimeError, match="CUDA"):
        train_run(tmp_path / "gpu", device="cuda")
    assert not (tmp_path / "gpu").exists()
