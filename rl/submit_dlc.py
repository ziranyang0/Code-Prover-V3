"""Render or explicitly submit a configured Code-Prover DLC job."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import uuid
from pathlib import Path

if __package__:
    from .checkpoint_compat import validate_resume_checkpoint
    from .source_fingerprint import git_source_fingerprint
    from .sandbox import e2b_connection_env
    from .judge_config import add_judge_arguments, judge_options
else:
    from checkpoint_compat import validate_resume_checkpoint
    from source_fingerprint import git_source_fingerprint
    from sandbox import e2b_connection_env
    from judge_config import add_judge_arguments, judge_options

DLC = "/etc/dsw/runtime/export_bin/aliyun"
REGION = "ap-southeast-1"
ENDPOINT = "pai-dlc.ap-southeast-1.aliyuncs.com"
IMAGE = os.environ.get("CODEPROVER_RL_IMAGE", "")
REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get(
    "CODEPROVER_PROJECT_ROOT",
    str(next((parent for parent in REPO_ROOT.parents if (parent / "miles").is_dir()),
             REPO_ROOT.parent)),
)).expanduser().resolve()

# Retain the queue alias; provide deployment identifiers at runtime.
QUEUE_NAMES = ("rbe",)


def _deployment_config(args) -> dict[str, str]:
    settings = {}
    for field in ("workspace_id", "resource_id", "data_source_id", "mount_path"):
        value = getattr(args, field, None) or os.environ.get("DLC_" + field.upper(), "")
        if not value or not value.strip():
            flag = "--" + field.replace("_", "-")
            raise ValueError(f"{flag} or DLC_{field.upper()} is required")
        settings[field] = value.strip()
    if not Path(settings["mount_path"]).is_absolute():
        raise ValueError("--mount-path must be absolute")
    return settings


# Example GPU-8 worker shape for the supported campaign topologies.
RESOURCE_CONFIG = {
    "CPU": "192",
    "GPU": "8",
    "GPUType": "",
    "Memory": "1800Gi",
    "SharedMemory": "1800Gi",
}
ACTOR_TOPOLOGY_BY_NODES = {
    8: {"world_size": 32, "expert_model_parallel_size": 4, "rollout_gpus": 32},
    10: {"world_size": 32, "expert_model_parallel_size": 4, "rollout_gpus": 48},
}


def _validate_key_file(path: Path) -> None:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError("E2B key must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
        raise ValueError("E2B key must be owner-owned with mode 0600")


def _git_patch_sha256(path: Path) -> str:
    return git_source_fingerprint(path)


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _path_entry(path: Path, kind: str, mount_root: Path) -> dict:
    exists = path.exists()
    type_ok = exists and (path.is_file() if kind == "file" else path.is_dir())
    return {
        "path": str(path),
        "kind": kind,
        "exists": exists,
        "type_ok": type_ok,
        "visible_from_data_mount": _under(path, mount_root),
    }


def inspect_runtime_paths(args) -> dict:
    queue = _deployment_config(args)
    mount_root = Path(queue["mount_path"]).resolve()
    codeprover = Path(args.codeprover_root).resolve()
    miles = Path(args.miles_root).resolve()
    megatron_lm = Path(args.megatron_lm_root).resolve()
    hf = Path(args.hf_checkpoint).resolve()
    megatron_checkpoint = Path(args.megatron_checkpoint).resolve()
    run_root = Path(args.run_root).resolve()
    prompt_data = Path(
        getattr(args, "prompt_data", None)
        or codeprover / "rl" / "data" / "trainset_problems_300.jsonl"
    ).resolve()
    task_root = Path(
        getattr(args, "task_root", None)
        or codeprover / "tasks" / "trainset_problems_300"
    ).resolve()
    checks = {
        "codeprover_launcher": _path_entry(codeprover / "rl" / "run_dlc.sh", "file", mount_root),
        "prompt_data": _path_entry(prompt_data, "file", mount_root),
        "task_root": _path_entry(task_root, "directory", mount_root),
        "miles_entrypoint": _path_entry(miles / "train_async.py", "file", mount_root),
        "megatron_lm_root": _path_entry(megatron_lm, "directory", mount_root),
        "hf_config": _path_entry(hf / "config.json", "file", mount_root),
        "megatron_tracker": _path_entry(
            megatron_checkpoint / "latest_checkpointed_iteration.txt", "file", mount_root
        ),
        "e2b_key_file": _path_entry(Path(args.e2b_key_file).resolve(), "file", mount_root),
    }
    if ca_bundle := getattr(args, "e2b_ca_bundle", None):
        checks["e2b_ca_bundle"] = _path_entry(Path(ca_bundle).resolve(), "file", mount_root)
    if load_checkpoint_dir := getattr(args, "load_checkpoint_dir", None):
        load_checkpoint_dir = Path(load_checkpoint_dir).resolve()
        checks["load_checkpoint_tracker"] = _path_entry(
            load_checkpoint_dir / "latest_checkpointed_iteration.txt", "file", mount_root
        )
    if debug_rollout := getattr(args, "load_debug_rollout_data", None):
        checks["load_debug_rollout_data"] = _path_entry(
            Path(debug_rollout.format(rollout_id=0)).resolve(), "file", mount_root
        )
    if eval_summary := getattr(args, "previous_stage_eval_summary", None):
        checks["previous_stage_eval_summary"] = _path_entry(
            Path(eval_summary).resolve(), "file", mount_root
        )
    judge = judge_options(args, default_artifacts=str(run_root / (getattr(args, "run_id", None) or args.name) / "proofs"),
                          validate=not bool(getattr(args, "load_debug_rollout_data", None)))
    if judge["judge_mode"] != "legacy" and not getattr(args, "load_debug_rollout_data", None):
        if judge["comparator_queue_dir"]:
            checks["comparator_queue"] = _path_entry(Path(judge["comparator_queue_dir"]).resolve(), "directory", mount_root)
        proof_dir = Path(judge["proof_artifacts_dir"]).resolve()
        proof_ancestor = proof_dir
        while not proof_ancestor.exists() and proof_ancestor != proof_ancestor.parent:
            proof_ancestor = proof_ancestor.parent
        checks["proof_artifacts_dir"] = {
            "path": str(proof_dir), "kind": "writable_output_directory",
            "ancestor_writable": proof_ancestor.is_dir() and os.access(proof_ancestor, os.W_OK),
            "visible_from_data_mount": _under(proof_dir, mount_root),
        }
    ancestor = run_root
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    checks["run_root"] = {
        "path": str(run_root),
        "kind": "writable_output_directory",
        "exists": run_root.is_dir(),
        "existing_ancestor": str(ancestor),
        "ancestor_writable": ancestor.is_dir() and os.access(ancestor, os.W_OK),
        "visible_from_data_mount": _under(run_root, mount_root),
    }
    return checks


def _missing_submit_inputs(checks: dict) -> list[str]:
    missing = []
    for name, entry in checks.items():
        if entry["kind"] == "writable_output_directory":
            valid = entry["ancestor_writable"] and entry["visible_from_data_mount"]
        else:
            valid = entry["exists"] and entry["type_ok"] and entry["visible_from_data_mount"]
        if not valid:
            missing.append(name)
    return missing


def build_body(args) -> dict:
    if not args.name.startswith("TRACES_Verification_RL_"):
        raise ValueError("DLC display name must start with TRACES_Verification_RL_")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise ValueError("DLC display name contains unsupported characters")
    if args.nodes not in ACTOR_TOPOLOGY_BY_NODES:
        raise ValueError("Singapore RBE RL requires 8 or 10 eight-GPU H200 nodes")
    if getattr(args, "submit", False) and not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", args.image):
        raise ValueError("--image must identify the built amd64 RL image by digest")
    codeprover_root = Path(args.codeprover_root).resolve()
    patch_sha256 = args.codeprover_patch_sha256 or _git_patch_sha256(codeprover_root)
    if not re.fullmatch(r"[0-9a-f]{64}", patch_sha256):
        raise ValueError("Code-Prover patch SHA-256 must be 64 lowercase hex characters")
    miles_root = Path(args.miles_root).resolve()
    miles_patch_sha256 = args.miles_patch_sha256 or _git_patch_sha256(miles_root)
    if not re.fullmatch(r"[0-9a-f]{64}", miles_patch_sha256):
        raise ValueError("Miles patch SHA-256 must be 64 lowercase hex characters")
    key_file = Path(args.e2b_key_file).resolve()
    prompt_data = Path(
        getattr(args, "prompt_data", None)
        or codeprover_root / "rl" / "data" / "trainset_problems_300.jsonl"
    ).resolve()
    task_root = Path(
        getattr(args, "task_root", None)
        or codeprover_root / "tasks" / "trainset_problems_300"
    ).resolve()
    for path, description in (
        (codeprover_root / "rl" / "run_dlc.sh", "Code-Prover DLC launcher"),
        (miles_root / "train_async.py", "official Miles train_async.py"),
        (key_file, "E2B key file"),
    ):
        if not path.exists():
            raise ValueError(f"missing {description}: {path}")
    _validate_key_file(key_file)
    queue = _deployment_config(args)
    run_id = getattr(args, "run_id", None) or args.name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("RL run id contains unsupported characters")
    attempt_id = getattr(args, "attempt_id", None) or args.name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_id):
        raise ValueError("DLC attempt id contains unsupported characters")
    connection = e2b_connection_env(
        getattr(args, "e2b_api_url", "") or "",
        getattr(args, "e2b_domain", "") or "",
        getattr(args, "e2b_sandbox_provider", "") or "",
    )
    if not args.e2b_template:
        raise ValueError("--e2b-template must name a template built on that service")
    metadata_json = getattr(args, "e2b_create_metadata_json", None)
    if metadata_json:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
        ):
            raise ValueError("E2B create metadata must contain string values")
        connection["E2B_CREATE_METADATA_JSON"] = json.dumps(metadata, separators=(",", ":"))
    profile = getattr(args, "e2b_bootstrap_profile", "prover")
    if profile not in ("prover", "lean-base"):
        raise ValueError("E2B bootstrap profile must be prover or lean-base")
    connection["E2B_BOOTSTRAP_PROFILE"] = profile
    create_timeout = getattr(args, "e2b_create_timeout_sec", 120)
    import math
    if not math.isfinite(create_timeout) or create_timeout <= 0:
        raise ValueError("E2B create timeout must be positive and finite")
    connection["E2B_CREATE_TIMEOUT_SEC"] = str(create_timeout)
    pool_size = getattr(args, "e2b_warm_pool_size", 0)
    create_concurrency = getattr(args, "e2b_create_concurrency", 8)
    pool_wait_timeout = getattr(args, "e2b_warm_pool_wait_timeout_sec", 1200)
    if pool_size < 0 or create_concurrency < 1:
        raise ValueError("E2B warm pool size must be nonnegative and creation concurrency positive")
    if not math.isfinite(pool_wait_timeout) or pool_wait_timeout <= 0:
        raise ValueError("E2B warm pool wait timeout must be positive and finite")
    connection["E2B_WARM_POOL_SIZE"] = str(pool_size)
    connection["E2B_CREATE_CONCURRENCY"] = str(create_concurrency)
    connection["E2B_WARM_POOL_WAIT_TIMEOUT_SEC"] = str(pool_wait_timeout)
    ca_bundle = getattr(args, "e2b_ca_bundle", None)
    if ca_bundle:
        ca_bundle = Path(ca_bundle).resolve()
        if not ca_bundle.is_file():
            raise ValueError("E2B CA bundle must be a readable PEM file")
        import ssl
        ssl.create_default_context(cafile=str(ca_bundle))
        connection["SSL_CERT_FILE"] = str(ca_bundle)
    connection["E2B_VALIDATE_API_KEY"] = (
        "true" if getattr(args, "e2b_validate_api_key", True) else "false"
    )
    judge = judge_options(
        args, default_artifacts=str(Path(args.run_root).resolve() / run_id / "proofs"),
        validate=not bool(getattr(args, "load_debug_rollout_data", None)),
    )
    envs = {
        **{"PROVER_" + key.upper(): str(value) for key, value in judge.items()},
        **connection,
        "RUN_ID": run_id,
        "ATTEMPT_ID": attempt_id,
        "RUN_ROOT": str(Path(args.run_root).resolve()),
        "CODEPROVER_ROOT": str(codeprover_root),
        "MILES_ROOT": str(miles_root),
        "MEGATRON_LM_ROOT": str(Path(args.megatron_lm_root).resolve()),
        "CODEPROVER_SOURCE_COMMIT": args.codeprover_commit,
        "CODEPROVER_PATCH_SHA256": patch_sha256,
        "MILES_SOURCE_COMMIT": args.miles_commit,
        "MILES_PATCH_SHA256": miles_patch_sha256,
        "HF_CHECKPOINT": str(Path(args.hf_checkpoint).resolve()),
        "MEGATRON_CHECKPOINT": str(Path(args.megatron_checkpoint).resolve()),
        "PROMPT_DATA": str(prompt_data),
        "TASK_ROOT": str(task_root),
        "E2B_TEMPLATE_ID": args.e2b_template,
        "E2B_API_KEY_FILE": str(key_file),
        "NNODES": str(args.nodes),
        "GPUS_PER_NODE": RESOURCE_CONFIG["GPU"],
        "NUM_ROLLOUT": str(args.num_rollout),
        "START_ROLLOUT_ID": str(getattr(args, "start_rollout_id", 0)),
        "ROLLOUT_BATCH_SIZE": str(args.rollout_batch_size),
        "N_SAMPLES_PER_PROMPT": str(args.n_samples_per_prompt),
        "GLOBAL_BATCH_SIZE": str(args.global_batch_size),
        "OVER_SAMPLING_MULTIPLIER": str(args.over_sampling_multiplier),
        "OVER_SAMPLING_BATCH_SIZE": str(
            args.rollout_batch_size * args.over_sampling_multiplier
        ),
        "DYNAMIC_SAMPLING_FILTER_PATH": (
            getattr(
                args,
                "dynamic_sampling_filter_path",
                "rl.filters.check_clean_and_nonzero_std",
            )
            if args.zero_group_filtering
            else ""
        ),
        "PROVER_MAX_TOTAL_TOKENS": str(args.prover_max_total_tokens),
        "PROVER_MAX_TURNS": str(args.prover_max_turns),
        "PROVER_MAX_TOKENS_PER_TURN": str(args.prover_max_tokens_per_turn),
        "PROVER_MAX_TRUNCATION_NUDGES": str(args.prover_max_truncation_nudges),
        "PROVER_MAX_TOOL_RESULT_TOKENS": str(args.prover_max_tool_result_tokens),
        "PROVER_WALL_TIME_BUDGET_SEC": str(args.prover_wall_time_budget_sec),
        "PROVER_EPISODE_TIMEOUT_SEC": str(args.prover_episode_timeout_sec),
        "PROVER_ROUTER_TIMEOUT_SEC": str(args.prover_router_timeout_sec),
        "PROVER_SANDBOX_CONCURRENCY": str(args.prover_sandbox_concurrency),
        "MAX_SEQ_LEN": str(args.max_seq_len),
        "ROLLOUT_MAX_RESPONSE_LEN": str(args.rollout_max_response_len),
        "MAX_TOKENS_PER_GPU": str(args.max_tokens_per_gpu),
        "LOG_PROBS_CHUNK_SIZE": str(args.log_probs_chunk_size),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_SERVER_CONCURRENCY": str(args.sglang_server_concurrency),
        "PROVER_ASYNC_QUEUE_SIZE": str(args.prover_async_queue_size),
        "PROVER_ASYNC_PAUSE_TIMEOUT_SEC": str(args.prover_async_pause_timeout_sec),
        "ROUTER_BALANCE_ABS_THRESHOLD": str(args.router_balance_abs_threshold),
        "SAVE_INTERVAL": str(getattr(args, "save_interval", 1)),
        "SAVE_RETAIN_INTERVAL": str(
            getattr(args, "save_retain_interval", None) or ""
        ),
        "OPTIMIZER_OFFLOAD_FRACTION": str(
            getattr(args, "optimizer_offload_fraction", 1.0)
        ),
        "NO_LOAD_OPTIM": "1" if getattr(args, "no_load_optim", False) else "0",
        "SAVE_OPTIM": "1" if getattr(args, "save_optim", False) else "0",
        "RESET_ROLLOUT_DATA_STATE": (
            "1" if getattr(args, "reset_rollout_data_state", False) else "0"
        ),
        "RAY_NUM_CPUS_PER_NODE": RESOURCE_CONFIG["CPU"],
        "NCCL_MNNVL_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_ALGO": "^NVLS",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCHDYNAMO_DISABLE": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "PYTHONUNBUFFERED": "1",
    }
    if debug_rollout := getattr(args, "load_debug_rollout_data", None):
        envs["LOAD_DEBUG_ROLLOUT_DATA"] = str(Path(debug_rollout).resolve())
    if load_checkpoint_dir := getattr(args, "load_checkpoint_dir", None):
        envs["LOAD_CHECKPOINT_DIR"] = str(Path(load_checkpoint_dir).resolve())
    return {
        "DisplayName": args.name,
        "JobType": "PyTorchJob",
        "WorkspaceId": queue["workspace_id"],
        "ResourceId": queue["resource_id"],
        "Priority": 9,
        "Accessibility": "PRIVATE",
        "JobMaxRunningTimeMinutes": args.max_running_minutes,
        "SuccessPolicy": "AllWorkers",
        "UserCommand": f"bash {codeprover_root}/rl/run_dlc.sh",
        "DataSources": [{
            "DataSourceId": queue["data_source_id"],
            "DataSourceVersion": "v1",
            "MountPath": queue["mount_path"],
        }],
        "Envs": envs,
        "JobSpecs": [{
            "Type": "Worker",
            "Image": args.image,
            "PodCount": args.nodes,
            "ResourceConfig": dict(RESOURCE_CONFIG),
            "RestartPolicy": "Never",
        }],
        "Settings": {
            "DisableEcsStockCheck": False,
            "EnableRDMA": True,
            "Tags": {
                "project": "codeprover",
                "framework": "miles",
                "train_mode": "async",
                "sandbox": "e2b",
                "nodes": str(args.nodes),
                "gpus_per_node": RESOURCE_CONFIG["GPU"],
                **({"owner": args.owner} if getattr(args, "owner", "") else {}),
                **({"topic_id": args.topic_id} if getattr(args, "topic_id", "") else {}),
            },
        },
    }


def _job_file_text(body: dict) -> str:
    spec = body["JobSpecs"][0]
    resource = spec["ResourceConfig"]
    data_source = body["DataSources"][0]
    envs = ",".join(f"{key}={value}" for key, value in sorted(body["Envs"].items()))
    tags = ",".join(f"{key}={value}" for key, value in sorted(body["Settings"]["Tags"].items()))
    values = {
        "name": body["DisplayName"],
        "workers": spec["PodCount"],
        "worker_image": spec["Image"],
        "worker_cpu": resource["CPU"],
        "worker_gpu": resource["GPU"],
        "worker_memory": resource["Memory"],
        "worker_shared_memory": resource["SharedMemory"],
        "command": body["UserCommand"],
        "workspace_id": body["WorkspaceId"],
        "resource_id": body["ResourceId"],
        "priority": body["Priority"],
        "accessibility": body["Accessibility"],
        "success_policy": body["SuccessPolicy"],
        "job_max_running_time_minutes": body["JobMaxRunningTimeMinutes"],
        "data_sources": (
            f"{data_source['DataSourceId']}:{data_source['DataSourceVersion']}:"
            f"{data_source['MountPath']}:"
        ),
        "envs": envs,
        "tags": tags,
        "advanced_settings": "EnableRDMA=true",
    }
    for key, value in values.items():
        if "\n" in str(value) or "\x00" in str(value):
            raise ValueError(f"unsupported newline/NUL in DLC parameter {key}")
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _submit_command(body: dict) -> list[str]:
    """Use the SG-authorized aliyun profile and send the complete API body.

    The preinstalled dlc binary uses a different injected identity. Sending
    JSON also preserves Settings.EnableRDMA, omitted by the old flag renderer.
    """
    return [
        DLC, "pai-dlc", "CreateJob",
        "--region", REGION,
        "--endpoint", ENDPOINT,
        "--body", json.dumps(body),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument(
        "--image", default=IMAGE, required=not bool(IMAGE),
        help="Built amd64 RL image; submission requires a registry URI pinned by digest",
    )
    parser.add_argument(
        "--run-id",
        help="Shared run directory id; set this to a previous run id to resume its checkpoint",
    )
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--stage", type=int, choices=range(1, 6))
    parser.add_argument(
        "--batch-dir",
        default=(
            f"{PROJECT_ROOT}/Megatron-LM-test-upstream/Code-Prover-V3/rl/data/"
            "d3-pool-v1/rl-batches-v1"
        ),
    )
    parser.add_argument("--start-rollout-id", type=int, default=0)
    parser.add_argument("--num-rollout", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--save-retain-interval", type=int)
    parser.add_argument("--optimizer-offload-fraction", type=float, default=1.0)
    parser.add_argument("--no-load-optim", action="store_true")
    parser.add_argument("--save-optim", action="store_true")
    parser.add_argument("--reset-rollout-data-state", action="store_true")
    parser.add_argument(
        "--load-checkpoint-dir",
        help=(
            "Checkpoint root to load; defaults to "
            "<run-root>/<run-id-or-name>/checkpoints"
        ),
    )
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--n-samples-per-prompt", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--over-sampling-multiplier", type=int, default=1)
    parser.add_argument("--zero-group-filtering", action="store_true")
    parser.add_argument(
        "--dynamic-sampling-filter-path",
        default="rl.filters.check_clean_and_nonzero_std",
    )
    parser.add_argument("--prover-max-total-tokens", type=int, default=32768)
    parser.add_argument("--prover-max-turns", type=int, default=64)
    parser.add_argument("--prover-max-tokens-per-turn", type=int, default=2048)
    parser.add_argument("--prover-max-truncation-nudges", type=int, default=1)
    add_judge_arguments(parser)
    parser.add_argument("--prover-max-tool-result-tokens", type=int, default=4096)
    parser.add_argument("--prover-wall-time-budget-sec", type=int, default=1200)
    parser.add_argument("--prover-episode-timeout-sec", type=int, default=2400)
    parser.add_argument("--prover-router-timeout-sec", type=int, default=180)
    parser.add_argument("--prover-sandbox-concurrency", type=int)
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--rollout-max-response-len", type=int, default=32768)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=32768)
    parser.add_argument("--log-probs-chunk-size", type=int, default=256)
    parser.add_argument("--sglang-server-concurrency", type=int, default=2)
    parser.add_argument("--prover-async-queue-size", type=int)
    parser.add_argument("--prover-async-pause-timeout-sec", type=int, default=300)
    parser.add_argument("--router-balance-abs-threshold", type=int, default=1)
    parser.add_argument("--queue", choices=QUEUE_NAMES, default="rbe")
    parser.add_argument("--topic-id", default=os.environ.get("DLC_TOPIC_ID", ""))
    parser.add_argument("--owner", default=os.environ.get("DLC_JOB_OWNER", ""))
    for field in ("workspace_id", "resource_id", "data_source_id", "mount_path"):
        parser.add_argument("--" + field.replace("_", "-"),
                            default=os.environ.get("DLC_" + field.upper(), ""))
    parser.add_argument("--codeprover-root", required=True)
    parser.add_argument("--miles-root", required=True)
    parser.add_argument("--codeprover-commit", required=True)
    parser.add_argument("--codeprover-patch-sha256")
    parser.add_argument("--miles-commit", default="12d80fea77d40087724a6910b08a0013a8b34d48")
    parser.add_argument("--miles-patch-sha256")
    parser.add_argument("--e2b-key-file", required=True)
    parser.add_argument("--e2b-api-url", default="", help="E2B-compatible HTTPS API origin")
    parser.add_argument("--e2b-domain", default="", help="Sandbox domain on the same service")
    parser.add_argument("--e2b-ca-bundle", help="PEM trust bundle visible on all DLC nodes")
    parser.add_argument("--e2b-create-metadata-json", help="Non-secret provider creation metadata")
    parser.add_argument("--e2b-create-timeout-sec", type=float, default=120)
    parser.add_argument("--e2b-create-concurrency", type=int, default=8)
    parser.add_argument("--e2b-warm-pool-size", type=int, default=0,
                        help="Fresh idle sandboxes maintained by the single rollout worker; 0 disables")
    parser.add_argument("--e2b-warm-pool-wait-timeout-sec", type=float, default=1200)
    parser.add_argument("--e2b-bootstrap-profile", choices=["prover", "lean-base"], default="prover")
    parser.add_argument("--e2b-validate-api-key", action=argparse.BooleanOptionalAction, default=True,
                        help="SDK key format validation; custom services may issue UUID keys")
    parser.add_argument("--e2b-sandbox-provider", choices=["aliyun"], default="",
                        help="Send X-Sandbox-Provider on gateway API requests")
    parser.add_argument("--prompt-data")
    parser.add_argument("--task-root")
    parser.add_argument("--load-debug-rollout-data")
    parser.add_argument("--previous-stage-eval-summary")
    parser.add_argument(
        "--e2b-template",
        default=os.environ.get("E2B_TEMPLATE_ID", ""),
        required=not bool(os.environ.get("E2B_TEMPLATE_ID")),
    )
    parser.add_argument(
        "--hf-checkpoint",
        default=f"{PROJECT_ROOT}/models/sft-hf",
    )
    parser.add_argument(
        "--megatron-checkpoint",
        default=(
            f"{PROJECT_ROOT}/models/"
            "sft-miles-torch-dist"
        ),
    )
    parser.add_argument("--megatron-lm-root", default=f"{PROJECT_ROOT}/Megatron-LM-test-upstream")
    parser.add_argument("--run-root", default=f"{PROJECT_ROOT}/rl-runs/codeprover-d3-miles")
    parser.add_argument("--output-dir", default=f"{PROJECT_ROOT}/rl-runs/dlc-submit")
    parser.add_argument("--max-running-minutes", type=int, default=480)
    parser.add_argument("--attempt-id")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.name is None:
        args.name = f"TRACES_Verification_RL_D3_stage{args.stage or 1:02d}"
    if args.nodes not in ACTOR_TOPOLOGY_BY_NODES:
        parser.error("--nodes must be 8 or 10 for the RBE H200 campaign")
    if args.stage is not None:
        args.rollout_batch_size = 32
        args.n_samples_per_prompt = 8
        args.global_batch_size = 256
        args.over_sampling_multiplier = 2
        args.zero_group_filtering = True
        args.save_interval = 5
        args.save_retain_interval = 1_000_000
        args.optimizer_offload_fraction = 0.0
        args.prover_sandbox_concurrency = 64
        args.prover_async_queue_size = 64
        args.hf_checkpoint = (
            f"{PROJECT_ROOT}/models/sft-hf"
        )
        args.megatron_checkpoint = (
            f"{PROJECT_ROOT}/models/"
            "sft-miles-torch-dist"
        )
        args.megatron_lm_root = f"{PROJECT_ROOT}/Megatron-LM-test-upstream"
        args.run_root = f"{PROJECT_ROOT}/rl-runs/codeprover-d3-miles"
        args.task_root = f"{PROJECT_ROOT}/Code-Prover-V3/data"
        args.max_running_minutes = max(args.max_running_minutes, 2880)
        args.prompt_data = str(
            Path(args.batch_dir) / f"batch-{args.stage:02d}.prompts.jsonl"
        )
        args.start_rollout_id = (args.stage - 1) * 30
        args.num_rollout = args.stage * 30
        args.save_optim = True
        args.reset_rollout_data_state = args.stage > 1
        args.no_load_optim = args.stage == 1
        if args.stage == 1:
            args.load_checkpoint_dir = args.megatron_checkpoint
        else:
            args.load_checkpoint_dir = str(
                (
                    Path(args.run_root)
                    / (args.run_id or args.name)
                    / "checkpoints"
                ).resolve()
            )
            if args.previous_stage_eval_summary is None:
                args.previous_stage_eval_summary = str(
                    Path(
                        f"{PROJECT_ROOT}/Megatron-LM-test-upstream/eval-results/"
                        f"rl-d3/{args.run_id or args.name}/stage-{args.stage - 1:02d}/summary.json"
                    )
                )
    expected_global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt
    if args.global_batch_size is None:
        args.global_batch_size = expected_global_batch_size
    elif args.global_batch_size != expected_global_batch_size:
        parser.error(
            "--global-batch-size must equal --rollout-batch-size * "
            "--n-samples-per-prompt"
        )
    if args.over_sampling_multiplier < 1:
        parser.error("--over-sampling-multiplier must be >= 1")
    if args.start_rollout_id < 0 or args.start_rollout_id >= args.num_rollout:
        parser.error("--start-rollout-id must be in [0, --num-rollout)")
    if args.save_interval < 1:
        parser.error("--save-interval must be >= 1")
    if args.save_retain_interval is not None and (
        args.save_retain_interval < 1
        or args.save_retain_interval % args.save_interval != 0
    ):
        parser.error("--save-retain-interval must be positive and divisible by --save-interval")
    if not 0.0 <= args.optimizer_offload_fraction <= 1.0:
        parser.error("--optimizer-offload-fraction must be in [0, 1]")
    over_sampling_batch_size = args.rollout_batch_size * args.over_sampling_multiplier
    if args.prover_async_queue_size is None:
        args.prover_async_queue_size = over_sampling_batch_size
    elif args.prover_async_queue_size < over_sampling_batch_size:
        parser.error(
            "--prover-async-queue-size must be >= the over-sampling batch size"
        )
    if args.prover_max_total_tokens < 1:
        parser.error("--prover-max-total-tokens must be >= 1")
    if args.prover_async_pause_timeout_sec < 1:
        parser.error("--prover-async-pause-timeout-sec must be >= 1")
    if args.prover_max_turns < 1:
        parser.error("--prover-max-turns must be >= 1")
    if args.prover_router_timeout_sec < 1:
        parser.error("--prover-router-timeout-sec must be >= 1")
    if args.prover_wall_time_budget_sec < 1:
        parser.error("--prover-wall-time-budget-sec must be >= 1")
    if args.prover_episode_timeout_sec <= args.prover_wall_time_budget_sec:
        parser.error(
            "--prover-episode-timeout-sec must exceed --prover-wall-time-budget-sec"
        )
    if args.prover_sandbox_concurrency is None:
        args.prover_sandbox_concurrency = (
            ACTOR_TOPOLOGY_BY_NODES[args.nodes]["rollout_gpus"] * 2
        )
    elif args.prover_sandbox_concurrency < 1:
        parser.error("--prover-sandbox-concurrency must be >= 1")
    if args.prover_max_tokens_per_turn < 1:
        parser.error("--prover-max-tokens-per-turn must be >= 1")
    if args.prover_max_truncation_nudges < 0:
        parser.error("--prover-max-truncation-nudges must be >= 0")
    if args.prover_max_tool_result_tokens < 128:
        parser.error("--prover-max-tool-result-tokens must be >= 128")
    if args.prover_max_tokens_per_turn > args.prover_max_total_tokens:
        parser.error("--prover-max-tokens-per-turn must not exceed --prover-max-total-tokens")
    if args.max_seq_len < args.prover_max_total_tokens:
        parser.error("--max-seq-len must be >= --prover-max-total-tokens")
    if args.rollout_max_response_len < args.prover_max_total_tokens:
        parser.error(
            "--rollout-max-response-len must be >= --prover-max-total-tokens"
        )
    if args.max_tokens_per_gpu < 1:
        parser.error("--max-tokens-per-gpu must be >= 1")
    if args.log_probs_chunk_size < 1:
        parser.error("--log-probs-chunk-size must be >= 1")
    if not 1 <= args.sglang_server_concurrency <= 16:
        parser.error("--sglang-server-concurrency must be in [1, 16]")
    if args.router_balance_abs_threshold < 0:
        parser.error("--router-balance-abs-threshold must be >= 0")
    if args.attempt_id is None:
        args.attempt_id = f"{args.name}-{uuid.uuid4().hex[:12]}"
    if args.load_checkpoint_dir is None:
        args.load_checkpoint_dir = str(
            (
                Path(args.run_root)
                / (args.run_id or args.name)
                / "checkpoints"
            ).resolve()
        )

    checks = inspect_runtime_paths(args)
    missing = _missing_submit_inputs(checks)
    resume_checkpoint = None
    if args.load_checkpoint_dir and checks["load_checkpoint_tracker"]["type_ok"]:
        actor_topology = ACTOR_TOPOLOGY_BY_NODES[args.nodes]
        try:
            resume_checkpoint = validate_resume_checkpoint(
                args.load_checkpoint_dir,
                actor_world_size=actor_topology["world_size"],
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=actor_topology["expert_model_parallel_size"],
                optimizer_offload_fraction=args.optimizer_offload_fraction,
                no_load_optim=args.no_load_optim,
                megatron_lm_root=args.megatron_lm_root,
            )
        except (ImportError, OSError) as exc:
            if args.submit:
                raise
            resume_checkpoint = {"validated": False, "error": str(exc)}
    body = build_body(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    body_path = output_dir / f"{args.name}.body.json"
    paths_path = output_dir / f"{args.name}.paths.json"
    job_file_path = output_dir / f"{args.name}.params"
    body_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    job_file_path.write_text(_job_file_text(body), encoding="utf-8")
    print(json.dumps({
        "body": str(body_path),
        "job_file": str(job_file_path),
        "missing_submit_inputs": missing,
        "paths": str(paths_path),
        "resume_checkpoint": resume_checkpoint,
        "image_pinned": bool(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", args.image)),
        "submit": args.submit,
    }, sort_keys=True))
    if not args.submit:
        return 0
    if missing:
        raise ValueError(f"refusing DLC submission; invalid runtime paths: {', '.join(missing)}")

    result = subprocess.run(
        _submit_command(body),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    job_ids = re.findall(r"\bdlc[a-z0-9]+\b", result.stdout)
    receipt = {"job_id": job_ids[0] if job_ids else None, "stdout": result.stdout, "stderr": result.stderr}
    receipt_path = output_dir / f"{args.name}.submit.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"job_id": receipt["job_id"], "receipt": str(receipt_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
