"""Render one GPU job with an explicit image digest and unique run name."""

import argparse
import re
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=("e101", "e102"), required=True)
    parser.add_argument("--image", required=True, help="registry/repository@sha256:...")
    parser.add_argument("--run-id", required=True, help="unique lowercase run identifier")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?", args.run_id):
        parser.error("run-id must be 1-40 lowercase letters, digits or internal hyphens")
    if not re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", args.image):
        parser.error("image must be pinned with @sha256:<64 hex digits>")
    root = Path(__file__).resolve().parents[1]
    rendered = subprocess.check_output(
        ["kubectl", "kustomize", str(root / "deploy" / "k8s" / args.node)], text=True
    )
    documents = list(yaml.safe_load_all(rendered))
    for document in documents:
        if document["kind"] == "Job":
            document["metadata"]["name"] += "-" + args.run_id
            document["spec"]["template"]["spec"]["containers"][0]["image"] = args.image
    print(yaml.safe_dump_all(documents, sort_keys=False), end="")


if __name__ == "__main__":
    main()
