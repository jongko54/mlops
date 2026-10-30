"""Render immutable source + a bounded e101-only CH4/5 integration Job."""

import argparse
import hashlib
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", required=True)
args = parser.parse_args()
data = {p.name: p.read_text() for p in (root / "src/mlops_lab").glob("*.py")}
data["requirements-platform.txt"] = (root / "requirements-platform.txt").read_text()
digest = hashlib.sha256(yaml.safe_dump(data).encode()).hexdigest()
cm_name = f"platform-source-{digest[:12]}"
namespace = "mlops-lab"
cm = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {"name": cm_name, "namespace": namespace},
    "immutable": True,
    "data": data,
}
pvc = {
    "apiVersion": "v1",
    "kind": "PersistentVolumeClaim",
    "metadata": {"name": "platform-artifacts-e101", "namespace": namespace},
    "spec": {
        "accessModes": ["ReadWriteOnce"],
        "storageClassName": "ceph-rbd",
        "resources": {"requests": {"storage": "5Gi"}},
    },
}
job = yaml.safe_load((root / "deploy/k8s/base/job.yaml").read_text())
job["metadata"] = {
    "name": args.name,
    "namespace": namespace,
    "annotations": {"mlops-lab/source-sha256": digest},
}
pod = job["spec"]["template"]["spec"]
pod["nodeSelector"] = {"kubernetes.io/hostname": "e101"}
job["spec"]["template"]["metadata"]["labels"] = {"app.kubernetes.io/name": "mlops-platform"}
container = pod["containers"][0]
container["image"] = (
    "pytorch/pytorch@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69"
)
container["command"] = ["/bin/bash", "-euc"]
container["args"] = [
    "python -m pip install --disable-pip-version-check --no-cache-dir "
    "--target /tmp/deps -r /opt/lab/mlops_lab/requirements-platform.txt; "
    "python -m pip freeze > /artifacts/${POD_NAME}-packages.txt; "
    'python -m mlops_lab.platform_lab --device cuda --output "/artifacts/${POD_NAME}"'
]
container["env"] += [
    {"name": "PYTHONPATH", "value": "/tmp/deps:/opt/lab"},
    {"name": "HOME", "value": "/tmp"},
    {"name": "FEAST_USAGE", "value": "False"},
    {"name": "MLFLOW_ENABLE_TELEMETRY", "value": "false"},
]
container["resources"]["requests"]["memory"] = "2Gi"
container["resources"]["limits"]["memory"] = "6Gi"
container["volumeMounts"] += [{"name": "source", "mountPath": "/opt/lab/mlops_lab"}]
pod["volumes"][0]["persistentVolumeClaim"]["claimName"] = "platform-artifacts-e101"
pod["volumes"][1]["emptyDir"]["sizeLimit"] = "2Gi"
pod["volumes"] += [{"name": "source", "configMap": {"name": cm_name}}]
print(yaml.safe_dump_all([cm, pvc, job], sort_keys=False))
