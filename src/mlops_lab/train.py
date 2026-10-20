"""Train, validate and record a tiny handwritten-digit classifier."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


def prepare_data(seed):
    data = load_digits()
    indices = np.arange(len(data.target))
    train, held_out = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=data.target
    )
    validation, test = train_test_split(
        held_out, test_size=0.5, random_state=seed, stratify=data.target[held_out]
    )
    return {
        name: {
            "indices": rows,
            "x": torch.from_numpy((data.data[rows] / 16.0).astype(np.float32)),
            "y": torch.from_numpy(data.target[rows].astype(np.int64)),
        }
        for name, rows in (("train", train), ("validation", validation), ("test", test))
    }


def make_model():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def accuracy(model, split, device):
    model.eval()
    with torch.inference_mode():
        predictions = model(split["x"].to(device)).argmax(dim=1)
        return float((predictions == split["y"].to(device)).float().mean().item())


def git_revision():
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_run(output, *, seed=42, epochs=30, device="cpu", min_accuracy=0.90):
    if epochs < 1 or not 0 <= min_accuracy <= 1:
        raise ValueError("epochs must be positive and min_accuracy must be in [0, 1]")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is disabled")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    splits = prepare_data(seed)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    x = splits["train"]["x"].to(device)
    y = splits["train"]["y"].to(device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(x), generator=generator).split(64):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
    validation_accuracy = accuracy(model, splits["validation"], device)
    passed = validation_accuracy >= min_accuracy
    # The test set is never used to choose weights or decide model promotion.
    test_accuracy = accuracy(model, splits["test"], device) if passed else None
    data = load_digits()
    dataset_bytes = data.data.astype("<f4").tobytes() + data.target.astype("<i8").tobytes()
    split_bytes = b"".join(
        splits[name]["indices"].astype("<i8").tobytes() for name in ("train", "validation", "test")
    )
    if device == "cuda":
        torch.cuda.synchronize()
    metrics = {
        "status": "passed" if passed else "rejected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": 0.01,
        "min_accuracy": min_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "split_sizes": {name: len(split["indices"]) for name, split in splits.items()},
        "dataset": "sklearn.load_digits",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": git_revision(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "elapsed_seconds": round(time.monotonic() - started, 4),
    }
    if passed:
        torch.save(
            {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "metrics": metrics},
            output / "model.pt",
        )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def predict(checkpoint, pixels):
    """Predict a batch of raw 64-pixel vectors with values in [0, 16]."""
    values = np.asarray(pixels, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 64 or len(values) == 0:
        raise ValueError("pixels must have shape (N, 64), with N > 0")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 16:
        raise ValueError("pixels must contain finite values in [0, 16]")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(values / 16.0)).argmax(dim=1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    args = parser.parse_args()
    try:
        metrics = train_run(
            args.output,
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
            min_accuracy=args.min_accuracy,
        )
    except (ValueError, RuntimeError, FileExistsError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0 if metrics["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
