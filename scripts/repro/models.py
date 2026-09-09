"""Download only immutable model revisions and verify upstream file hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]


def verified(path, entry):
    if not path.is_file() or (entry["size"] is not None and path.stat().st_size != entry["size"]):
        return False
    algorithm = hashlib.sha256() if entry["sha256"] else hashlib.sha1()
    if not entry["sha256"]:
        algorithm.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            algorithm.update(block)
    return algorithm.hexdigest() == (entry["sha256"] or entry["git_blob"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=["base"])
    args = parser.parse_args()
    spec = json.loads((ROOT / "deployment/repro/manifest.json").read_text())
    records = {}
    for name, model in spec["models"].items():
        if model["profile"] not in args.profiles:
            continue
        destination = ROOT / ".repro/models" / model["directory"]
        print(f"Downloading {model['repo_id']}@{model['revision']}", flush=True)
        pending = [entry["path"] for entry in model["files"] if not verified(destination / entry["path"], entry)]
        reused = len(model["files"]) - len(pending)
        if pending:
            snapshot_download(repo_id=model["repo_id"], revision=model["revision"], local_dir=destination,
                              allow_patterns=pending, max_workers=4, token=os.environ.get("HF_TOKEN") or False)
        hashes = {}
        for entry in model["files"]:
            path = destination / entry["path"]
            if not verified(path, entry):
                raise RuntimeError(f"Missing/incorrect model file: {name}/{entry['path']}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            value = digest.hexdigest()
            if entry["sha256"] and value != entry["sha256"]:
                raise RuntimeError(f"Model checksum mismatch: {name}/{entry['path']}")
            hashes[entry["path"]] = value
        records[name] = {"revision": model["revision"], "files": hashes, "verified_cache_files": reused,
                         "requested_download_files": len(pending)}
        print(f"Verified {name}: {len(hashes)} files", flush=True)
    (ROOT / ".repro/models-verified.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
