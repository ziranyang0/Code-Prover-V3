"""Fail-fast validation for official-Miles async RL with an E2B backend."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from rl.checkpoint_compat import validate_resume_checkpoint
from rl.judge_config import add_judge_arguments, preflight_judge
from rl.provenance import validate_prompt_data
from rl.sandbox import E2B_SDK_VERSION, e2b_connection_env, load_e2b_api_key
from rl.source_fingerprint import git_source_fingerprint

OFFICIAL_MILES_COMMIT = "12d80fea77d40087724a6910b08a0013a8b34d48"


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _git_patch_sha256(path: Path) -> str:
    return git_source_fingerprint(path)


def _require_clean_git(path: Path, description: str) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if status:
        raise ValueError(f"{description} source is dirty; use an immutable clean worktree")


def _require(path: Path, description: str) -> None:
    if not path.exists():
        raise ValueError(f"missing {description}: {path}")


def validate(args) -> dict:
    judge = preflight_judge(args)
    repo = Path(args.repo_root).resolve()
    miles = Path(args.miles_root).resolve()
    megatron_lm = Path(args.megatron_lm_root).resolve()
    hf = Path(args.hf_checkpoint).resolve()
    megatron = Path(args.megatron_checkpoint).resolve()
    prompt_data = Path(args.prompt_data).resolve()
    task_root = Path(args.task_root).resolve()

    supported_topologies = {
        (1, 8): (4, 4),
        (2, 4): (4, 4),
        (4, 4): (4, 12),
        (6, 4): (8, 16),
        (16, 4): (32, 32),
        (8, 8): (32, 32),
        (10, 8): (32, 48),
        (2, 8): (8, 8),
    }
    if getattr(args, "training_only", False):
        supported_topologies[(4, 8)] = (32, 0)
    topology = (args.nodes, args.gpus_per_node)
    if topology not in supported_topologies:
        raise ValueError(
            "supported async topologies are 1x8, 2x4, 4x4, 6x4, 16x4, 8x8, 10x8, or 2x8"
        )
    expected_actor_gpus, expected_rollout_gpus = supported_topologies[topology]
    if (args.actor_gpus, args.rollout_gpus) != (expected_actor_gpus, expected_rollout_gpus):
        raise ValueError(
            f"{args.nodes}-node topology requires actor_gpus={expected_actor_gpus}, "
            f"rollout_gpus={expected_rollout_gpus}"
        )
    for path, description in (
        (repo / "rl" / "fully_async_rollout.py", "V3 async rollout module"),
        (repo / "rl" / "generate_with_prover.py", "V3 Miles adapter"),
        (miles / "train_async.py", "official Miles train_async.py"),
        (miles / "scripts" / "models" / "qwen3.6-35B-A3B.sh", "Miles model profile"),
        (megatron_lm / "megatron" / "core", "Megatron-LM Python package"),
        (hf / "config.json", "Hugging Face config"),
        (megatron / "latest_checkpointed_iteration.txt", "Megatron tracker"),
        (prompt_data, "prompt data"),
        (task_root, "task root"),
    ):
        _require(path, description)

    miles_head = _git_head(miles)
    expected_miles_commit = args.expected_miles_commit or OFFICIAL_MILES_COMMIT
    if miles_head != expected_miles_commit:
        raise ValueError(
            f"Miles HEAD mismatch: {miles_head} != {expected_miles_commit}"
        )
    miles_patch_sha256 = _git_patch_sha256(miles)
    if (
        args.expected_miles_patch_sha256
        and miles_patch_sha256 != args.expected_miles_patch_sha256
    ):
        raise ValueError(
            "Miles patch SHA-256 mismatch: "
            f"{miles_patch_sha256} != {args.expected_miles_patch_sha256}"
        )
    repo_head = _git_head(repo)
    if args.expected_repo_commit and repo_head != args.expected_repo_commit:
        raise ValueError(f"V3 HEAD mismatch: {repo_head} != {args.expected_repo_commit}")
    repo_patch_sha256 = _git_patch_sha256(repo)
    if (
        args.expected_repo_patch_sha256
        and repo_patch_sha256 != args.expected_repo_patch_sha256
    ):
        raise ValueError(
            "V3 patch SHA-256 mismatch: "
            f"{repo_patch_sha256} != {args.expected_repo_patch_sha256}"
        )
    if not args.allow_dirty_source:
        _require_clean_git(repo, "V3")
        _require_clean_git(miles, "Miles")

    if not args.allow_missing_e2b:
        load_e2b_api_key()
        if not args.e2b_template:
            raise ValueError("E2B template is empty")
        actual_e2b = importlib.metadata.version("e2b")
        if actual_e2b != E2B_SDK_VERSION:
            raise ValueError(f"e2b=={E2B_SDK_VERSION} required; found {actual_e2b}")

    prompt_count = validate_prompt_data(prompt_data, task_root)
    if args.dynamic_sampling_filter_path:
        module_name, function_name = args.dynamic_sampling_filter_path.rsplit(".", 1)
        dynamic_filter = getattr(importlib.import_module(module_name), function_name)
        if not callable(dynamic_filter):
            raise ValueError("dynamic sampling filter is not callable")
    tracker = (megatron / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip()
    if tracker != "release" and not tracker.isdecimal():
        raise ValueError(f"invalid Megatron tracker value: {tracker!r}")

    resume_checkpoint = None
    if args.load_checkpoint_dir:
        resume_checkpoint = validate_resume_checkpoint(
            args.load_checkpoint_dir,
            actor_world_size=args.actor_gpus,
            tensor_model_parallel_size=args.actor_tensor_parallel_size,
            pipeline_model_parallel_size=args.actor_pipeline_parallel_size,
            context_parallel_size=args.actor_context_parallel_size,
            expert_model_parallel_size=args.actor_expert_parallel_size,
            optimizer_offload_fraction=args.optimizer_offload_fraction,
            no_load_optim=args.no_load_optim,
            megatron_lm_root=megatron_lm,
        )

    return {
        "ok": True,
        "judge": judge,
        "repo_commit": repo_head,
        "repo_patch_sha256": repo_patch_sha256,
        "miles_commit": miles_head,
        "miles_patch_sha256": miles_patch_sha256,
        "nodes": args.nodes,
        "gpus_per_node": args.gpus_per_node,
        "actor_gpus": args.actor_gpus,
        "rollout_gpus": args.rollout_gpus,
        "prompt_count": prompt_count,
        "dynamic_sampling_filter_path": args.dynamic_sampling_filter_path or None,
        "checkpoint_tracker": tracker,
        "resume_checkpoint": resume_checkpoint,
        "sandbox": "e2b",
        "e2b_connection": e2b_connection_env(
            os.environ.get("E2B_API_URL", ""), os.environ.get("E2B_DOMAIN", ""),
            os.environ.get("E2B_SANDBOX_PROVIDER", ""),
        ),
        "e2b_template": args.e2b_template or "<local-preflight-skipped>",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--miles-root", required=True)
    parser.add_argument("--megatron-lm-root", required=True)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--megatron-checkpoint", required=True)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--dynamic-sampling-filter-path", default="")
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--actor-gpus", type=int, required=True)
    parser.add_argument("--rollout-gpus", type=int, required=True)
    parser.add_argument("--actor-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--actor-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--actor-context-parallel-size", type=int, default=1)
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--actor-expert-parallel-size", type=int, default=4)
    parser.add_argument("--load-checkpoint-dir")
    parser.add_argument("--optimizer-offload-fraction", type=float, default=1.0)
    parser.add_argument("--no-load-optim", action="store_true")
    parser.add_argument("--e2b-template", default="")
    parser.add_argument("--expected-repo-commit", default="")
    parser.add_argument("--expected-repo-patch-sha256", default="")
    parser.add_argument("--expected-miles-commit", default="")
    parser.add_argument("--expected-miles-patch-sha256", default="")
    parser.add_argument("--allow-missing-e2b", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    add_judge_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(validate(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
