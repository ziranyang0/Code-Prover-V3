"""Shared judge settings for rollouts, evaluation, and deployment preflight."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

DEFAULT_JUDGE_MODE = "comparator"
JUDGE_DEFAULTS = {
    "judge_mode": DEFAULT_JUDGE_MODE,
    "comparator_image": "",
    "comparator_queue_dir": "",
    "comparator_timeout_sec": 1200,
    "comparator_concurrency": 2,
    "proof_artifacts_dir": "",
}


def add_judge_arguments(parser):
    for name, default in JUDGE_DEFAULTS.items():
        kwargs = {"default": os.environ.get("PROVER_" + name.upper(), default),
                  "type": type(default)}
        if name == "judge_mode":
            kwargs["choices"] = ["comparator", "shadow", "legacy"]
        parser.add_argument("--prover-" + name.replace("_", "-"), **kwargs)


def validate_judge_options(options):
    mode = options["judge_mode"]
    if mode not in ("comparator", "shadow", "legacy"):
        raise ValueError(f"unknown prover judge mode: {mode}")
    if mode == "legacy":
        return
    image, queue = options["comparator_image"], options["comparator_queue_dir"]
    if bool(image) == bool(queue):
        raise ValueError("comparator requires exactly one of --prover-comparator-image or --prover-comparator-queue-dir")
    if not options["proof_artifacts_dir"]:
        raise ValueError("comparator requires --prover-proof-artifacts-dir on durable storage")
    for name in ("proof_artifacts_dir", "comparator_queue_dir"):
        if options[name] and not Path(options[name]).is_absolute():
            raise ValueError(f"--prover-{name.replace('_', '-')} must be absolute")
    if min(options["comparator_timeout_sec"], options["comparator_concurrency"]) <= 0:
        raise ValueError("comparator timeout and concurrency must be positive")


def judge_options(args, *, default_artifacts="", validate=True):
    options = {key: getattr(args, "prover_" + key, default)
               for key, default in JUDGE_DEFAULTS.items()}
    options["proof_artifacts_dir"] = options["proof_artifacts_dir"] or default_artifacts
    if validate:
        validate_judge_options(options)
    return options


def preflight_judge(args):
    if getattr(args, "training_only", False):
        return {"skipped": True, "reason": "training_only"}
    options = judge_options(args)
    if options["judge_mode"] == "legacy":
        return options
    if options["comparator_queue_dir"]:
        from verifier.comparator.queue import healthy
        service = healthy(Path(options["comparator_queue_dir"]))
        options["service"] = {key: service[key] for key in ("image_id", "catalog_sha256") if key in service}
    else:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", options["comparator_image"]],
            check=True, capture_output=True, text=True, timeout=30,
        )
        options["image_id"] = result.stdout.strip()
    artifacts = Path(options["proof_artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    # Actually open/write on the mounted filesystem; permissions alone are insufficient.
    with tempfile.TemporaryFile(dir=artifacts) as probe:
        probe.write(b"comparator-preflight\n")
        probe.flush()
        os.fsync(probe.fileno())
    return options
