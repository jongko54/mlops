"""CH4/5 integration lab; optional dependencies are loaded only by lab operations."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PIXELS = [f"pixel_{i}" for i in range(64)]


def feature_matrix(rows, expected_ids):
    """Validate and align Feast records before feeding the model."""
    try:
        ids = list(rows["sample_id"])
        if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
            raise ValueError("feature entity IDs differ from requested IDs")
        matrix = np.asarray([rows[column] for column in PIXELS], dtype=np.float32).T
        if matrix.shape != (len(ids), 64) or not np.isfinite(matrix).all():
            raise ValueError("missing or invalid feature values")
        if matrix.min() < 0 or matrix.max() > 1:
            raise ValueError("features must already be normalized to [0, 1]")
        return matrix[[ids.index(i) for i in expected_ids]]
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete feature schema") from exc


def prepare_features(root):
    """Persist normalized pixels and exercise Feast point-in-time/online retrieval."""
    import pandas as pd
    import yaml
    from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
    from feast.types import Float32
    from sklearn.datasets import load_digits

    from mlops_lab.train import prepare_data

    repo = root / "feast"
    repo.mkdir()
    data = load_digits()
    event_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    query_time = event_time + timedelta(hours=1)
    frame = pd.DataFrame((data.data / 16).astype(np.float32), columns=PIXELS)
    frame["sample_id"] = np.arange(len(frame), dtype=np.int64)
    frame["event_timestamp"] = event_time
    # A deliberately different future version must not leak into historical lookup.
    future = frame.iloc[[0]].copy()
    future["event_timestamp"] = event_time + timedelta(days=1)
    future["pixel_0"] = np.float32(1.0)
    source_path = repo / "pixels.parquet"
    pd.concat([frame, future], ignore_index=True).to_parquet(source_path, index=False)
    config = {
        "project": "digits_lab",
        "provider": "local",
        "registry": str(repo / "registry.db"),
        "offline_store": {"type": "file"},
        "online_store": {"type": "sqlite", "path": str(repo / "online.db")},
        "entity_key_serialization_version": 3,
    }
    (repo / "feature_store.yaml").write_text(yaml.safe_dump(config))
    store = FeatureStore(repo_path=str(repo))
    entity = Entity(name="sample", join_keys=["sample_id"], value_type=ValueType.INT64)
    view = FeatureView(
        name="pixels",
        entities=[entity],
        ttl=timedelta(days=7),
        schema=[Field(name=column, dtype=Float32) for column in PIXELS],
        source=FileSource(path=str(source_path), timestamp_field="event_timestamp"),
        online=True,
    )
    store.apply([entity, view])
    refs = [f"pixels:{column}" for column in PIXELS]
    historic = store.get_historical_features(
        entity_df=pd.DataFrame({"sample_id": [0], "event_timestamp": [query_time]}),
        features=refs,
    ).to_df()
    np.testing.assert_array_equal(feature_matrix(historic, [0])[0], frame.loc[0, PIXELS])
    store.materialize(event_time - timedelta(seconds=1), query_time)
    ids = prepare_data(42)["test"]["indices"].tolist()
    entity_rows = [{"sample_id": int(i)} for i in ids]
    online = store.get_online_features(features=refs, entity_rows=entity_rows).to_dict()
    np.testing.assert_array_equal(feature_matrix(online, ids), (data.data[ids] / 16))
    request = {
        "sample_ids": ids,
        "expected_labels": data.target[ids].tolist(),
        "query_time": query_time.isoformat(),
        "synthetic_timestamps": True,
    }
    (root / "request.json").write_text(json.dumps(request))
    evidence = {
        "point_in_time_future_excluded": True,
        "online_values_match_source": True,
        "features": 64,
        "test_rows": len(ids),
        "offline_store": "Parquet",
        "online_store": "SQLite",
    }
    (root / "feast-checks.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def train_and_register(root, device):
    import mlflow
    import mlflow.pytorch
    import torch
    from mlflow.models import infer_signature

    from mlops_lab.train import make_model, prepare_data, train_run

    tracking_uri = f"sqlite:///{root}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment("digits-ch04-ch05")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        metrics = train_run(root / "training", device=device)
        mlflow.log_params({k: metrics[k] for k in ("seed", "epochs", "batch_size", "device")})
        mlflow.set_tags({k: metrics[k] for k in ("dataset_sha256", "split_sha256", "code_sha256")})
        mlflow.log_metrics({"validation_accuracy": metrics["validation_accuracy"]})
        mlflow.log_artifact(str(root / "training" / "metrics.json"))
        if metrics["status"] != "passed":
            raise RuntimeError("validation rejected; no model registration")
        mlflow.log_metric("test_accuracy", metrics["test_accuracy"])
        model = make_model()
        saved = torch.load(root / "training" / "model.pt", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        sample = prepare_data(42)["validation"]["x"][:5]
        with torch.inference_mode():
            signature = infer_signature(sample.numpy(), model(sample).numpy())
        info = mlflow.pytorch.log_model(
            model,
            name="digits",
            registered_model_name="digits-mlp-ch45",
            signature=signature,
            serialization_format="pickle",
            pip_requirements=["torch==2.8.0", "numpy==2.2.6"],
        )
        version = info.registered_model_version
        model_uri = f"models:/digits-mlp-ch45/{version}"
        reloaded = mlflow.pytorch.load_model(model_uri)
        with torch.inference_mode():
            torch.testing.assert_close(reloaded(sample), model(sample))
        result = {
            "tracking_uri": tracking_uri,
            "run_id": run.info.run_id,
            "model_uri": model_uri,
            "model_version": str(version),
            "registered_model_reload_matches": True,
            "metrics": metrics,
        }
    (root / "model-record.json").write_text(json.dumps(result, indent=2))
    return result


def fetch_request(path, output):
    request = json.loads(Path(path).read_text())
    if not request["sample_ids"] or len(set(request["sample_ids"])) != len(request["sample_ids"]):
        raise ValueError("request IDs must be nonempty and unique")
    Path(output).write_text(json.dumps(request))


def retrieve_batch(repo, request_path, output):
    import pandas as pd
    from feast import FeatureStore

    request = json.loads(Path(request_path).read_text())
    ids = request["sample_ids"]
    store = FeatureStore(repo_path=repo)
    rows = store.get_historical_features(
        entity_df=pd.DataFrame(
            {"sample_id": ids, "event_timestamp": pd.to_datetime(request["query_time"])}
        ),
        features=[f"pixels:{column}" for column in PIXELS],
    ).to_df()
    matrix = feature_matrix(rows, ids)
    np.savez(output, pixels=matrix, ids=ids, labels=request["expected_labels"])


def infer_batch(tracking_uri, model_uri, feature_path, output):
    import mlflow
    import mlflow.pytorch
    import torch

    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pytorch.load_model(model_uri)
    batch = np.load(feature_path)
    with torch.inference_mode():
        predictions = model(torch.from_numpy(batch["pixels"])).argmax(1).numpy()
    result = {
        "sample_ids": batch["ids"].tolist(),
        "predictions": predictions.tolist(),
        "correct": int((predictions == batch["labels"]).sum()),
        "total": len(predictions),
        "accuracy": float((predictions == batch["labels"]).mean()),
        "model_uri": model_uri,
    }
    Path(output).write_text(json.dumps(result, indent=2))


def write_results(source, destination):
    result = json.loads(Path(source).read_text())
    if len(result["predictions"]) != result["total"]:
        raise ValueError("incomplete batch output")
    Path(destination).write_text(json.dumps(result, indent=2))
    return result["accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Keep all MLflow artifacts under this run's persistent output directory.
    import os

    os.chdir(root)
    record = train_and_register(root, args.device)
    features = prepare_features(root)
    from mlops_lab.platform_pipeline import run_pipeline

    run_pipeline(root, record)
    batch = json.loads((root / "batch-predictions.json").read_text())
    if abs(batch["accuracy"] - record["metrics"]["test_accuracy"]) > 1e-6:
        raise RuntimeError("pipeline predictions disagree with held-out test accuracy")
    evidence = {
        "status": "passed",
        "kfp_execution": "SubprocessRunner inside one process environment; not KFP server",
        "model": record,
        "feast": features,
        "batch": {k: v for k, v in batch.items() if k not in ("predictions", "sample_ids")},
    }
    (root / "verification.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
