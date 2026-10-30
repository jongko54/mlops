"""Compile a four-component KFP pipeline and exercise it with the local runner.

Paths refer to a shared filesystem for this development lab. Remote KFP needs
an image containing mlops_lab plus remote services/object storage or PVC mounts.
"""

from kfp import compiler, dsl, local


@dsl.component(install_kfp_package=False)
def read_data(request_path: str, request: dsl.Output[dsl.Dataset]):
    from mlops_lab.platform_lab import fetch_request

    fetch_request(request_path, request.path)


@dsl.component(install_kfp_package=False)
def retrieve_features(
    repo_path: str, request: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]
):
    from mlops_lab.platform_lab import retrieve_batch

    # np.savez appends .npz for filenames; a binary file handle preserves KFP paths.
    with open(features.path, "wb") as output:
        retrieve_batch(repo_path, request.path, output)


@dsl.component(install_kfp_package=False)
def run_inference(
    tracking_uri: str,
    model_uri: str,
    features: dsl.Input[dsl.Dataset],
    predictions: dsl.Output[dsl.Dataset],
):
    from mlops_lab.platform_lab import infer_batch

    infer_batch(tracking_uri, model_uri, features.path, predictions.path)


@dsl.component(install_kfp_package=False)
def write_data(predictions: dsl.Input[dsl.Dataset], result_path: str) -> float:
    from mlops_lab.platform_lab import write_results

    return write_results(predictions.path, result_path)


@dsl.pipeline(name="digits-ch45-batch-inference")
def digits_pipeline(
    request_path: str, repo_path: str, tracking_uri: str, model_uri: str, result_path: str
):
    request = read_data(request_path=request_path)
    features = retrieve_features(repo_path=repo_path, request=request.outputs["request"])
    prediction = run_inference(
        tracking_uri=tracking_uri,
        model_uri=model_uri,
        features=features.outputs["features"],
    )
    write_data(predictions=prediction.outputs["predictions"], result_path=result_path)


def run_pipeline(root, model_record):
    compiler.Compiler().compile(digits_pipeline, str(root / "pipeline.yaml"))
    local.init(
        runner=local.SubprocessRunner(use_venv=False), pipeline_root=str(root / "kfp-outputs")
    )
    digits_pipeline(
        request_path=str(root / "request.json"),
        repo_path=str(root / "feast"),
        tracking_uri=model_record["tracking_uri"],
        model_uri=model_record["model_uri"],
        result_path=str(root / "batch-predictions.json"),
    )
