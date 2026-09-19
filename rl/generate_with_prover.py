"""miles custom-generate function for Code-Prover RL rollouts.

One episode = one sandbox (docker or E2B) + a token-level agent loop speaking
the qwen-native-v1 dialect against miles' sglang router, graded at the end by
the independent Comparator backend (Lean/Mathlib 4.28).

Wire-up (miles launch flags):
    --custom-generate-function-path rl.generate_with_prover.generate
    --prompt-data <jsonl from rl/make_prompt_data.py>
    --input-key prompt --metadata-key metadata
    (reward is computed inside `generate`, no --custom-rm-path needed)

The tool surface, spec guard, and truncation-nudge semantics are reused from
agents/qwen_native_agent.py so RL rollouts and harbor evals behave
identically; the only differences are token-level transport (sglang native
/generate with input_ids) and compaction being disabled (a context reset
would break the linear token stream — long episodes end as TRUNCATED).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path

# Ray rollout workers import this module by dotted path; make the repo root
# importable regardless of the worker's CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import qwen_native_v1 as codec  # noqa: E402
from agents.qwen_native_agent import (  # noqa: E402
    ALLOWED_TOOLS,
    TRUNCATION_NUDGE_PROMPT,
    QwenNativeAgent,
    _READONLY_TOOLS,
)
from rl.judge_config import (  # noqa: E402
    DEFAULT_JUDGE_MODE, JUDGE_DEFAULTS, add_judge_arguments, judge_options,
    validate_judge_options,
)
from rl.sandbox import create_sandbox  # noqa: E402
from rl.diagnostics import (  # noqa: E402
    EpisodeDiagnostics, current_episode, failure_context, operation, set_phase,
)
from rl.errors import safe_failure_detail as _safe_failure_detail  # noqa: E402
from rl.provenance import resolve_task_dir, task_provenance  # noqa: E402
from rl.token_stream import TokenStream  # noqa: E402

GRADE_TIMEOUT_SEC = 1320  # Task verifier allows 1200s of Lean; reserve 120s for result publication.
GRADE_UPLOAD_TIMEOUT_SEC = 300
GRADE_REWARD_TIMEOUT_SEC = 190  # Three 60s reads, backoff, and scheduling overhead.
EPISODE_SCHEDULING_MARGIN_SEC = 60
_sandbox_semaphore: asyncio.Semaphore | None = None
logger = logging.getLogger(__name__)


class RouterGenerateTimeout(RuntimeError):
    """A single SGLang router request exceeded its explicit timeout."""


class GenerationTimeout(RuntimeError):
    """Generation/tool execution exhausted its budget before verification."""


class EpisodeTimeout(RuntimeError):
    """The complete multi-turn prover episode exceeded its wall-time limit."""


@dataclasses.dataclass
class EpisodeConfig:
    model_path: str
    max_turns: int = 64
    max_total_tokens: int = 65536
    max_tokens_per_turn: int = 2048
    max_truncation_nudges: int = 1
    max_tool_result_tokens: int = 4096
    sandbox_backend: str = "docker"
    docker_image: str = "lizenan1995/code-prover-lean:latest"
    e2b_template: str = ""
    sandbox_cpus: float | None = None
    sandbox_memory_gb: float | None = None
    wall_time_budget_sec: int = 1200
    episode_timeout_sec: int = 2400
    router_timeout_sec: int = 180
    judge_mode: str = DEFAULT_JUDGE_MODE
    comparator_image: str = ""
    comparator_queue_dir: str = ""
    comparator_timeout_sec: int = 1200
    comparator_concurrency: int = 2
    proof_artifacts_dir: str = ""


@dataclasses.dataclass
class EpisodeResult:
    tokens: list[int]
    prompt_len: int
    loss_mask: list[int]
    logprobs: list[float]
    status: str                  # completed | truncated | failed
    reward: float
    rewards: dict
    n_turns: int
    stop_detail: str
    guard_events: list[dict] = dataclasses.field(default_factory=list)
    weight_versions: list[str] = dataclasses.field(default_factory=list)
    generation_time_sec: float = 0.0
    tool_time_sec: float = 0.0
    tool_calls: int = 0
    tool_result_truncations: int = 0
    model_generated_tokens: int = 0
    tool_and_feedback_tokens: int = 0
    synthetic_tokens: int = 0
    length_limited_turns: int = 0
    tito_prefix_checks: int = 0
    early_compile_probes: int = 0
    early_compile_successes: int = 0
    episode_wall_time_sec: float = 0.0
    judge_details: dict = dataclasses.field(default_factory=dict)

    @property
    def response_length(self) -> int:
        return len(self.tokens) - self.prompt_len


def _wrap_tool_responses(results: list[str]) -> str:
    # Must match codec.encode_messages flush_tool_results byte-for-byte.
    return "\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in results)


def _append_follow_up(ts: TokenStream, content: str, max_total_tokens: int) -> bool:
    """Append a complete injected turn, or do nothing when it cannot fit."""
    ids = ts.user_turn_tokens(content)
    if len(ts.tokens) + len(ids) > max_total_tokens:
        return False
    ts.tokens += ids
    ts.loss_mask += [0] * len(ids)
    ts.logprobs += [0.0] * len(ids)
    return True


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _cap_tool_result(tokenizer, result: str, max_tokens: int) -> tuple[str, int, int]:
    """Keep a bounded head/tail view of a tool result.

    Returns ``(text, original_tokens, kept_tokens)``. Character-boundary
    truncation avoids corrupting Unicode, while token-counting makes the
    injected-context limit independent of source-code/tokenizer density.
    """
    original_tokens = _token_count(tokenizer, result)
    if original_tokens <= max_tokens:
        return result, original_tokens, original_tokens

    marker = (
        "\n\n[... tool output truncated by RL context guard: "
        f"original_tokens={original_tokens} ...]\n\n"
    )
    if _token_count(tokenizer, marker) >= max_tokens:
        raise ValueError("max_tool_result_tokens is too small for truncation marker")

    # Maximize retained characters with a 3:1 head/tail split. The head keeps
    # command/file context; the tail usually contains the final Lean error.
    low, high = 0, len(result)
    best = marker
    best_tokens = _token_count(tokenizer, best)
    while low <= high:
        retained_chars = (low + high) // 2
        head_chars = (retained_chars * 3) // 4
        tail_chars = retained_chars - head_chars
        candidate = result[:head_chars] + marker
        if tail_chars:
            candidate += result[-tail_chars:]
        candidate_tokens = _token_count(tokenizer, candidate)
        if candidate_tokens <= max_tokens:
            best, best_tokens = candidate, candidate_tokens
            low = retained_chars + 1
        else:
            high = retained_chars - 1
    return best, original_tokens, best_tokens


_LEAN_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
_LEAN_LINE_COMMENT_RE = re.compile(r"--.*$", re.MULTILINE)
_LEAN_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_LEAN_PLACEHOLDER_RE = re.compile(r"\b(?:sorry|admit)\b")
_LEAN_FORBIDDEN_RE = re.compile(
    r"\b(?:axiom|constant|opaque|unsafe)\s+|\bnegation\b|Tacs\.Negate|negate_goal"
)


def _eligible_for_compile_early_stop(source: str) -> bool:
    """Conservative source gate before probing whether the proof is done.

    The real verifier remains authoritative.  This only avoids asking the
    model for more turns after a sorry-free, policy-clean file already compiles.
    """
    cleaned = _LEAN_BLOCK_COMMENT_RE.sub(
        lambda match: "\n" * match.group().count("\n"), source
    )
    cleaned = _LEAN_LINE_COMMENT_RE.sub("", cleaned)
    cleaned = _LEAN_STRING_RE.sub('""', cleaned)
    return not _LEAN_PLACEHOLDER_RE.search(cleaned) and not _LEAN_FORBIDDEN_RE.search(
        cleaned
    )


async def _compile_early_stop_probe(sandbox, task_file_target: str) -> tuple[bool, str]:
    """Compile the exact verifier target without exposing hidden grader files."""
    try:
        result = await sandbox.exec(
            f"cd /task && lake env lean {shlex.quote(task_file_target)}",
            timeout_sec=180,
        )
    except Exception as exc:  # a probe failure should not kill a valid episode
        return False, f"probe_exception:{type(exc).__name__}"
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    error = any(re.search(r"\berror\b\s*:", line) for line in output.splitlines())
    return result.return_code == 0 and not error, f"return_code={result.return_code},error={error}"


class _ToolSurface(QwenNativeAgent):
    """Reuses the eval agent's tool dispatch / spec guard / bridge setup
    without its chat-API run loop."""

    def __init__(self):
        super().__init__(logs_dir=Path(tempfile.mkdtemp(prefix="prover-rl-")))


async def run_episode(
    generate_fn,
    sandbox,
    instruction: str,
    task_dir: Path,
    cfg: EpisodeConfig,
) -> EpisodeResult:
    """Drive one full episode. `generate_fn(input_ids, remaining_tokens) -> dict` must return
    {"token_ids": [...], "logprobs": [...], "finish_reason": "stop"|"length",
     "text": str} for one assistant turn."""
    validate_judge_options({name: getattr(cfg, name) for name in JUDGE_DEFAULTS})
    set_phase("initialization")
    episode_started = time.monotonic()
    generation_time_sec = 0.0
    tool_time_sec = 0.0
    tool_calls = 0
    tool_result_truncations = 0
    model_generated_tokens = 0
    synthetic_tokens = 0
    length_limited_turns = 0
    tito_prefix_checks = 0
    early_compile_probes = 0
    early_compile_successes = 0
    tests_dir = task_dir / "tests"
    ts = TokenStream(cfg.model_path)
    ts.start(instruction)

    generation_timeout = cfg.wall_time_budget_sec + cfg.router_timeout_sec
    generation_deadline = asyncio.timeout(generation_timeout)
    try:
        async with generation_deadline:
            # RL sandboxes come from the GENERIC base image (no per-task docker
            # build), so the task's .lean file must be placed at the path the
            # verifier will grade — tests/task_file.txt is authoritative.
            task_file_target = (tests_dir / "task_file.txt").read_text(encoding="utf-8").strip()
            await sandbox.upload_file(task_dir / "environment" / "task.lean", task_file_target)

            surface = _ToolSurface()
            with operation("tool_surface.setup"):
                await surface.setup(sandbox)
            surface._guard_path = task_file_target
            surface._guard_original = await surface._read_task_file(sandbox, task_file_target)
            surface._guard_last_good = surface._guard_original
            guard_events: list[dict] = []
            weight_versions: list[str] = []

            def emit(kind: str, payload: dict) -> None:
                guard_events.append({"type": kind, **payload})

            status, stop_detail = "truncated", "max_turns"
            n_nudges = 0
            n_turns = 0
            probed_source_hashes: set[str] = set()
            for turn in range(1, cfg.max_turns + 1):
                if time.monotonic() - episode_started >= cfg.wall_time_budget_sec:
                    status, stop_detail = "truncated", "wall_time_budget"
                    break
                n_turns = turn
                remaining_tokens = cfg.max_total_tokens - len(ts.tokens)
                if remaining_tokens <= 1:
                    status, stop_detail = "truncated", "token_budget"
                    break

                # Reserve one token for a synthetic <|im_end|> when SGLang stops on
                # length. Never let a multi-turn request exceed the episode budget.
                request_prefix = tuple(ts.tokens)
                set_phase("generation", turn=turn)
                with operation("router.generate"):
                    out = await generate_fn(list(request_prefix), remaining_tokens - 1)
                if tuple(ts.tokens) != request_prefix:
                    raise RuntimeError("TITO invariant violated: generation mutated the request prefix")
                generation_time_sec += float(out.get("duration_sec", 0.0))
                # Keep one entry per request, including missing/invalid values.
                # Dropping them would make partial provenance look complete.
                weight_versions.append(str(out.get("weight_version")))
                token_ids = out["token_ids"]
                logprobs = out["logprobs"]
                if len(token_ids) != len(logprobs):
                    raise RuntimeError("router returned mismatched token_ids and logprobs")
                if len(token_ids) > remaining_tokens - 1:
                    raise RuntimeError("router exceeded the requested episode token budget")
                ts.append_generated(token_ids, logprobs)
                model_generated_tokens += len(token_ids)
                if tuple(ts.tokens[:len(request_prefix)]) != request_prefix:
                    raise RuntimeError("TITO invariant violated: sampled tokens rewrote the request prefix")
                if ts.tokens[len(request_prefix):] != token_ids:
                    raise RuntimeError("TITO invariant violated: stored output differs from sampled token ids")
                tito_prefix_checks += 1
                finish = out["finish_reason"]
                length_limited_turns += int(finish == "length")
                if not ts.tokens or ts.tokens[-1] != ts.glue.im_end_id:
                    # Turn cut mid-stream (length) or stop token trimmed upstream:
                    # close the turn with a synthetic <|im_end|> (mask 0) so the
                    # conversation format stays valid for any follow-up turn.
                    ts.tokens.append(ts.glue.im_end_id)
                    ts.loss_mask.append(0)
                    ts.logprobs.append(0.0)
                    synthetic_tokens += 1

                # Tool actions must be decoded from the exact sampled IDs stored for
                # training, never from a separately serialized server text field.
                assistant_text = ts.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                decoded = codec.decode_assistant(
                    content=assistant_text,
                    reasoning_content=None,
                    structured_tool_calls=None,
                    allowed_tool_names=ALLOWED_TOOLS,
                    call_id_namespace=f"turn{n_turns}",
                    accept_qwen_function_xml=True,
                )

                if decoded.errors and not decoded.tool_calls:
                    logger.warning(
                        "Code-Prover protocol decode failed turn=%s errors=%s "
                        "assistant_preview=%r",
                        n_turns,
                        [error.code for error in decoded.errors],
                        assistant_text[:500],
                    )
                    recovery = [
                        json.dumps(
                            codec.protocol_error_recovery_call(
                                err, index=i, call_id_namespace=f"turn{n_turns}"
                            ).arguments,
                            ensure_ascii=False,
                        )
                        for i, err in enumerate(decoded.errors)
                    ]
                    if not _append_follow_up(ts, _wrap_tool_responses(recovery), cfg.max_total_tokens):
                        status, stop_detail = "truncated", "token_budget"
                        break
                    continue

                if not decoded.tool_calls:
                    if finish == "length" and n_nudges < cfg.max_truncation_nudges:
                        n_nudges += 1
                        if not _append_follow_up(
                            ts, _wrap_tool_responses([TRUNCATION_NUDGE_PROMPT]), cfg.max_total_tokens
                        ):
                            status, stop_detail = "truncated", "token_budget"
                            break
                        continue
                    if finish == "length":
                        status, stop_detail = "truncated", "truncated_without_tool_call"
                    else:
                        status, stop_detail = "completed", "final_answer"
                    break
                n_nudges = 0

                results = []
                compile_early_stop = False
                for tc in decoded.tool_calls:
                    if time.monotonic() - episode_started >= cfg.wall_time_budget_sec:
                        status, stop_detail = "truncated", "wall_time_budget"
                        break
                    tool_started = time.monotonic()
                    set_phase("tool", turn=n_turns, tool=tc.name)
                    with operation("tool.dispatch"):
                        result = await surface._dispatch(sandbox, tc.name, tc.arguments)
                    if tc.name not in _READONLY_TOOLS:
                        result += await surface._guard_task_file(sandbox, emit)
                    tool_duration_sec = time.monotonic() - tool_started
                    tool_time_sec += tool_duration_sec
                    tool_calls += 1
                    original_chars = len(result)
                    result, original_tokens, kept_tokens = _cap_tool_result(
                        ts.tokenizer, result, cfg.max_tool_result_tokens
                    )
                    truncated = kept_tokens < original_tokens
                    tool_result_truncations += int(truncated)
                    logger.info(
                        "Code-Prover tool complete task=%s turn=%s tool=%s "
                        "duration_sec=%.3f original_chars=%s kept_chars=%s "
                        "original_tokens=%s kept_tokens=%s truncated=%s",
                        task_dir.name,
                        n_turns,
                        tc.name,
                        tool_duration_sec,
                        original_chars,
                        len(result),
                        original_tokens,
                        kept_tokens,
                        truncated,
                    )
                    guard_events.append({
                        "type": "tool_metric",
                        "turn": n_turns,
                        "name": tc.name,
                        "duration_sec": tool_duration_sec,
                        "original_tokens": original_tokens,
                        "kept_tokens": kept_tokens,
                        "truncated": truncated,
                    })
                    results.append(result)
                    if tc.name not in _READONLY_TOOLS:
                        current = await surface._read_task_file(sandbox, task_file_target)
                        if current is not None and _eligible_for_compile_early_stop(current):
                            source_hash = hashlib.sha256(current.encode()).hexdigest()
                            if source_hash not in probed_source_hashes:
                                probed_source_hashes.add(source_hash)
                                probe_started = time.monotonic()
                                passed, detail = await _compile_early_stop_probe(
                                    sandbox, task_file_target
                                )
                                probe_duration_sec = time.monotonic() - probe_started
                                tool_time_sec += probe_duration_sec
                                early_compile_probes += 1
                                early_compile_successes += int(passed)
                                guard_events.append({
                                    "type": "early_compile_probe",
                                    "turn": n_turns,
                                    "duration_sec": probe_duration_sec,
                                    "passed": passed,
                                    "detail": detail,
                                })
                                logger.info(
                                    "Code-Prover early compile probe task=%s turn=%s "
                                    "duration_sec=%.3f passed=%s detail=%s",
                                    task_dir.name,
                                    n_turns,
                                    probe_duration_sec,
                                    passed,
                                    detail,
                                )
                                if passed:
                                    status, stop_detail = "completed", "compile_passed"
                                    compile_early_stop = True
                                    break
                if compile_early_stop:
                    break
                if stop_detail == "wall_time_budget":
                    break
                if not _append_follow_up(ts, _wrap_tool_responses(results), cfg.max_total_tokens):
                    status, stop_detail = "truncated", "token_budget"
                    break

    except TimeoutError as exc:
        if not generation_deadline.expired():
            raise
        # Cancellation may leave a remote tool running. Fail this trajectory
        # and let generate() destroy its sandbox; never grade concurrently.
        raise GenerationTimeout(
            f"generation/tool phase exceeded {generation_timeout}s; verification reserve untouched"
        ) from exc

    ts.validate()
    if len(ts.tokens) > cfg.max_total_tokens:
        raise RuntimeError("episode token budget invariant violated")
    set_phase("verification")
    judge_details = {}
    with operation("verifier.grade"):
        if cfg.judge_mode == "legacy":
            reward, rewards = await _grade(sandbox, tests_dir)
        else:
            from rl.comparator_judge import grade_with_comparator
            reward, rewards, judge_details = await grade_with_comparator(sandbox, tests_dir, cfg, _grade)
    return EpisodeResult(
        tokens=ts.tokens,
        prompt_len=ts.prompt_len,
        loss_mask=ts.loss_mask,
        logprobs=ts.logprobs,
        status=status,
        reward=reward,
        rewards=rewards,
        judge_details=judge_details,
        n_turns=n_turns,
        stop_detail=stop_detail,
        guard_events=guard_events,
        weight_versions=weight_versions,
        generation_time_sec=generation_time_sec,
        tool_time_sec=tool_time_sec,
        tool_calls=tool_calls,
        tool_result_truncations=tool_result_truncations,
        model_generated_tokens=model_generated_tokens,
        tool_and_feedback_tokens=len(ts.tokens) - ts.prompt_len - model_generated_tokens - synthetic_tokens,
        synthetic_tokens=synthetic_tokens,
        length_limited_turns=length_limited_turns,
        tito_prefix_checks=tito_prefix_checks,
        early_compile_probes=early_compile_probes,
        early_compile_successes=early_compile_successes,
        episode_wall_time_sec=time.monotonic() - episode_started,
    )


async def _grade(sandbox, tests_dir: Path) -> tuple[float, dict]:
    set_phase("verifier_upload")
    async with asyncio.timeout(GRADE_UPLOAD_TIMEOUT_SEC):
        for f in tests_dir.iterdir():
            with operation("verifier.upload"):
                await sandbox.upload_file(f, f"/tests/{f.name}")
    set_phase("verification")
    verification_started = time.monotonic()
    async with asyncio.timeout(GRADE_TIMEOUT_SEC):
        with operation("verifier.exec"):
            verifier = await sandbox.exec("bash /tests/test.sh", timeout_sec=GRADE_TIMEOUT_SEC)
    logger.info("Code-Prover verification complete duration_sec=%.3f return_code=%s context=%s",
                time.monotonic() - verification_started, verifier.return_code,
                json.dumps(failure_context()))
    if verifier.return_code != 0:
        raise RuntimeError(
            f"verifier entrypoint failed with code {verifier.return_code}: "
            f"{verifier.stderr[-500:]}"
        )
    set_phase("reward_read")
    # Reading the published result is idempotent. Recover transient transport
    # failures here while the sandbox and generated proof are still available.
    # Never restart an execution whose completion state is unknown.
    async with asyncio.timeout(GRADE_REWARD_TIMEOUT_SEC):
        for attempt in range(3):
            try:
                with operation("verifier.read_reward"):
                    r = await sandbox.exec("cat /logs/verifier/reward.json", timeout_sec=60)
                if r.return_code != 0:
                    raise RuntimeError(f"verifier did not publish reward.json: {r.stderr[-500:]}")
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError("failed to read verifier reward.json after 3 attempts") from exc
                logger.warning("retrying verifier reward read attempt=%s/3 detail=%s",
                               attempt + 1, _safe_failure_detail(exc))
                await asyncio.sleep(1)
            else:
                break
    try:
        rewards = json.loads(r.stdout)
        if not isinstance(rewards, dict) or "reward" not in rewards:
            raise ValueError("reward.json must be an object containing reward")
        return float(rewards["reward"]), rewards
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError("verifier published an invalid reward.json") from exc


# --------------------------------------------------------------- miles glue --


def add_arguments(parser):
    parser.add_argument("--prover-model-path", type=str, required=True,
                        help="HF checkpoint dir (tokenizer + chat template)")
    parser.add_argument("--prover-task-root", type=str, required=True,
                        help="harbor task dataset root, e.g. tasks/trainset_problems_300")
    parser.add_argument("--prover-sandbox-backend", type=str, default="docker",
                        choices=["docker", "e2b"])
    parser.add_argument("--prover-docker-image", type=str,
                        default="lizenan1995/code-prover-lean:latest")
    add_judge_arguments(parser)
    parser.add_argument("--prover-e2b-template", type=str, default="")
    parser.add_argument("--prover-max-turns", type=int, default=64)
    parser.add_argument("--prover-max-total-tokens", type=int, default=65536)
    parser.add_argument("--prover-max-tokens-per-turn", type=int, default=2048)
    parser.add_argument("--prover-max-truncation-nudges", type=int, default=1)
    parser.add_argument("--prover-max-tool-result-tokens", type=int, default=4096)
    parser.add_argument("--prover-sandbox-concurrency", type=int, default=32)
    parser.add_argument("--prover-wall-time-budget-sec", type=int, default=1200)
    parser.add_argument("--prover-episode-timeout-sec", type=int, default=2400)
    parser.add_argument("--prover-router-timeout-sec", type=int, default=180)
    parser.add_argument("--prover-eval-prompt-data", type=str, default=None)
    parser.add_argument("--prover-full-eval-samples-per-prompt", type=int, default=8,
                        help="Samples per fixed evaluation task at the end of a training stage")
    parser.add_argument("--prover-tuning-prompt-data", type=str, default=None)


def weight_versions_complete(versions: list[str]) -> bool:
    """Every request must identify a published engine weight version."""
    return bool(versions) and all(
        isinstance(version, str) and version.isascii() and version.isdecimal()
        and int(version) > 0 for version in versions
    )


def weight_version_origin(args) -> int:
    """Map a process-local update counter to the training update sequence."""
    start = getattr(args, "start_rollout_id", 0)
    interval = getattr(args, "update_weights_interval", 1)
    if start is None or start < 0 or interval < 1:
        raise ValueError("weight version origin requires a nonnegative start and positive update interval")
    return start // interval


def effective_episode_timeout(args) -> int:
    # Generation and tools have a shared hard deadline. Verification receives
    # a separate reserve for execution, uploads, and bounded reward reads.
    if min(args.prover_wall_time_budget_sec, args.prover_router_timeout_sec,
           args.prover_episode_timeout_sec) <= 0:
        raise ValueError("generation, router and episode timeouts must be positive")
    minimum = (args.prover_wall_time_budget_sec + args.prover_router_timeout_sec
               + GRADE_UPLOAD_TIMEOUT_SEC + GRADE_TIMEOUT_SEC
               + GRADE_REWARD_TIMEOUT_SEC + EPISODE_SCHEDULING_MARGIN_SEC)
    if getattr(args, "prover_judge_mode", DEFAULT_JUDGE_MODE) != "legacy":
        minimum += getattr(args, "prover_comparator_timeout_sec", 1200) + 300
    return max(args.prover_episode_timeout_sec, minimum)


def _episode_config(args) -> EpisodeConfig:
    return EpisodeConfig(
        **judge_options(args),
        model_path=args.prover_model_path,
        max_turns=args.prover_max_turns,
        max_total_tokens=args.prover_max_total_tokens,
        max_tokens_per_turn=args.prover_max_tokens_per_turn,
        max_truncation_nudges=args.prover_max_truncation_nudges,
        max_tool_result_tokens=args.prover_max_tool_result_tokens,
        sandbox_backend=args.prover_sandbox_backend,
        docker_image=args.prover_docker_image,
        e2b_template=args.prover_e2b_template,
        wall_time_budget_sec=args.prover_wall_time_budget_sec,
        episode_timeout_sec=effective_episode_timeout(args),
        router_timeout_sec=args.prover_router_timeout_sec,
    )


def _router_generate_fn(
    args,
    sampling_params: dict,
    *,
    timeout_sec: int,
    max_tokens_per_turn: int,
    task_name: str,
    sample_index: int,
):
    from miles.utils.http_utils import post

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    call_index = 0

    async def generate_fn(input_ids: list[int], remaining_tokens: int) -> dict:
        nonlocal call_index
        call_index += 1
        turn_sampling_params = dict(sampling_params)
        turn_sampling_params["max_new_tokens"] = min(
            int(turn_sampling_params["max_new_tokens"]),
            remaining_tokens,
            max_tokens_per_turn,
        )
        requested_tokens = int(turn_sampling_params["max_new_tokens"])
        payload = {
            "input_ids": input_ids,
            "sampling_params": {**turn_sampling_params, "no_stop_trim": True},
            "return_logprob": True,
        }
        started = time.monotonic()
        logger.info(
            "Code-Prover router request start task=%s sample_index=%s call=%s "
            "input_tokens=%s remaining_tokens=%s max_new_tokens=%s timeout_sec=%s",
            task_name,
            sample_index,
            call_index,
            len(input_ids),
            remaining_tokens,
            requested_tokens,
            timeout_sec,
        )
        try:
            output = await asyncio.wait_for(post(url, payload), timeout=timeout_sec)
        except TimeoutError as exc:
            duration_sec = time.monotonic() - started
            logger.warning(
                "Code-Prover router request timeout task=%s sample_index=%s call=%s "
                "duration_sec=%.3f input_tokens=%s max_new_tokens=%s",
                task_name,
                sample_index,
                call_index,
                duration_sec,
                len(input_ids),
                requested_tokens,
            )
            raise RouterGenerateTimeout(
                f"router generation exceeded {timeout_sec}s"
            ) from exc
        duration_sec = time.monotonic() - started
        pairs = output["meta_info"]["output_token_logprobs"]
        finish_reason = output["meta_info"]["finish_reason"]["type"]
        logger.info(
            "Code-Prover router request complete task=%s sample_index=%s call=%s "
            "duration_sec=%.3f input_tokens=%s output_tokens=%s finish_reason=%s",
            task_name,
            sample_index,
            call_index,
            duration_sec,
            len(input_ids),
            len(pairs),
            finish_reason,
        )
        return {
            "token_ids": [p[1] for p in pairs],
            "logprobs": [p[0] for p in pairs],
            "finish_reason": finish_reason,
            "text": output["text"],
            "weight_version": output["meta_info"].get("weight_version"),
            "duration_sec": duration_sec,
        }

    return generate_fn


def _resolve_task(sample, task_root: Path) -> tuple[Path, str, dict]:
    meta = dict(sample.metadata or {})
    # Failure diagnostics belong to the preceding attempt, not a retry that
    # later succeeds. Keep infrastructure_retries for the group budget.
    for key in list(meta):
        if key.startswith("failure_") or key in ("infrastructure_failure", "sandbox_id"):
            meta.pop(key)
    task_dir = resolve_task_dir(meta, task_root)
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    if "instruction" in meta and meta["instruction"] != instruction:
        raise ValueError("sample metadata.instruction does not match task source")
    expected = task_provenance(task_dir)
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"sample metadata.{key} does not match task source")
    if isinstance(sample.prompt, str) and sample.prompt != instruction:
        # HF chat templates commonly apply Jinja's ``trim`` filter to message
        # content, so an authoritative instruction ending in a newline is
        # rendered without that final newline.  Keep the provenance guard, but
        # compare the exact non-padding content that can survive rendering.
        rendered_instruction = instruction.strip()
        if not rendered_instruction or rendered_instruction not in sample.prompt:
            raise ValueError("sample prompt does not contain task instruction.md")
    return task_dir, instruction, {**meta, **expected}


def _mark_failed(sample, meta: dict, exc: Exception):
    from miles.utils.types import Sample

    sample.response = sample.response or ""
    sample.response_length = 0
    sample.tokens = []
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.reward = 0.0
    sample.status = Sample.Status.FAILED
    sample.metadata = {
        **meta,
        "infrastructure_failure": True,
        "failure_type": type(exc).__name__,
        "failure_detail": _safe_failure_detail(exc, limit=4000),
        **failure_context(),
    }
    return sample


async def generate(input):
    """Current official Miles ``GenerateFnInput -> GenerateFnOutput`` entrypoint."""
    from miles.rollout.base_types import GenerateFnOutput
    from miles.utils.types import Sample

    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params

    global _sandbox_semaphore
    if _sandbox_semaphore is None:
        _sandbox_semaphore = asyncio.Semaphore(args.prover_sandbox_concurrency)

    cfg = _episode_config(args)
    try:
        task_dir, instruction, meta = _resolve_task(sample, Path(args.prover_task_root))
    except Exception as exc:  # source mismatch is an infrastructure failure, never a proof failure
        logger.warning(
            "Code-Prover task resolution failed sample_index=%s type=%s detail=%s",
            sample.index,
            type(exc).__name__,
            _safe_failure_detail(exc),
        )
        return GenerateFnOutput(samples=_mark_failed(sample, dict(sample.metadata or {}), exc))

    async with _sandbox_semaphore:
        sandbox = None
        sample_wall_started = time.monotonic()
        diagnostic = EpisodeDiagnostics(task_dir.name, sample.index)
        context_token = current_episode.set(diagnostic)
        try:
            sandbox = await create_sandbox(
                cfg.sandbox_backend,
                image=cfg.docker_image,
                template=cfg.e2b_template,
                cpus=cfg.sandbox_cpus,
                memory_gb=cfg.sandbox_memory_gb,
                timeout=max(3600, cfg.episode_timeout_sec + 300),
                metadata={
                    "purpose": "codeprover-rl",
                    "run_id": os.environ.get("RUN_ID", "unknown")[:100],
                    "task_name": task_dir.name[:100],
                    "sample_index": str(sample.index),
                },
            )
            diagnostic.sandbox_id = getattr(
                getattr(sandbox, "_sbx", None), "sandbox_id", getattr(sandbox, "_cid", None)
            )
            sandbox_acquire_time = time.monotonic() - sample_wall_started
            episode_deadline = asyncio.timeout(cfg.episode_timeout_sec)
            try:
                async with episode_deadline:
                    ep = await run_episode(
                        _router_generate_fn(
                            args,
                            sampling_params,
                            timeout_sec=cfg.router_timeout_sec,
                            max_tokens_per_turn=cfg.max_tokens_per_turn,
                            task_name=task_dir.name,
                            sample_index=sample.index,
                        ),
                        sandbox,
                        instruction,
                        task_dir,
                        cfg,
                    )
            except TimeoutError as exc:
                # An operation may raise its own TimeoutError before the outer
                # episode deadline. Preserve that cause instead of claiming
                # the whole episode exceeded its configured lifetime.
                if not episode_deadline.expired():
                    raise
                raise EpisodeTimeout(
                    f"prover episode exceeded {cfg.episode_timeout_sec}s"
                ) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced to async worker as retryable infra
            logger.warning(
                "Code-Prover episode infrastructure failure task=%s sample_index=%s "
                "duration_sec=%.3f context=%s type=%s detail=%s",
                task_dir.name,
                sample.index,
                time.monotonic() - sample_wall_started,
                json.dumps(failure_context(), ensure_ascii=False),
                type(exc).__name__,
                _safe_failure_detail(exc, limit=None),
            )
            return GenerateFnOutput(samples=_mark_failed(sample, meta, exc))
        finally:
            try:
                if sandbox is not None:
                    await sandbox.close()
            finally:
                current_episode.reset(context_token)

    sample.tokens = ep.tokens
    sample.response_length = ep.response_length
    sample.response = TokenStream(cfg.model_path).tokenizer.decode(
        ep.tokens[ep.prompt_len:]
    )
    sample.loss_mask = ep.loss_mask
    sample.rollout_log_probs = ep.logprobs
    sample.weight_versions = ep.weight_versions
    sample.reward = ep.reward
    sample_wall_time_sec = time.monotonic() - sample_wall_started
    sample.non_generation_time = max(
        sample_wall_time_sec - ep.generation_time_sec,
        0.0,
    )
    sample.status = (
        Sample.Status.COMPLETED if ep.status == "completed" else Sample.Status.TRUNCATED
    )
    sample.metadata = {
        **meta,
        "rewards": ep.rewards,
        "judge_details": ep.judge_details,
        "n_turns": ep.n_turns,
        "stop_detail": ep.stop_detail,
        "prompt_tokens": ep.prompt_len,
        "response_tokens": ep.response_length,
        "trained_tokens": sum(ep.loss_mask),
        "weight_versions_complete": weight_versions_complete(ep.weight_versions),
        "weight_version_origin": weight_version_origin(args),
        "weight_version_start_rollout_id": getattr(args, "start_rollout_id", 0),
        "weight_update_interval": getattr(args, "update_weights_interval", 1),
        "generation_time_sec": ep.generation_time_sec,
        "sandbox_acquire_time_sec": sandbox_acquire_time,
        "sandbox_pool_hit": bool(getattr(sandbox, "warm_pool_hit", False)),
        "sandbox_id": getattr(getattr(sandbox, "_sbx", None), "sandbox_id", None),
        "tool_time_sec": ep.tool_time_sec,
        "tool_calls": ep.tool_calls,
        "tool_result_truncations": ep.tool_result_truncations,
        "model_generated_tokens": ep.model_generated_tokens,
        "tool_and_feedback_tokens": ep.tool_and_feedback_tokens,
        "synthetic_tokens": ep.synthetic_tokens,
        "length_limited_turns": ep.length_limited_turns,
        "tito_prefix_checks": ep.tito_prefix_checks,
        "early_compile_probes": ep.early_compile_probes,
        "early_compile_successes": ep.early_compile_successes,
        "tito_session_mismatch": [],
        "guard_events": ep.guard_events,
        "non_generation_time_sec": sample.non_generation_time,
        "episode_wall_time_sec": sample_wall_time_sec,
    }
    sample.validate()
    logger.info(
        "Code-Prover episode complete task=%s sample_index=%s status=%s reward=%s "
        "turns=%s prompt_tokens=%s response_tokens=%s trained_tokens=%s "
        "generation_time_sec=%.3f tool_time_sec=%.3f tool_calls=%s "
        "tool_result_truncations=%s tito_prefix_checks=%s "
        "early_compile_probes=%s early_compile_successes=%s "
        "non_generation_time_sec=%.3f wall_time_sec=%.3f "
        "stop_detail=%s",
        task_dir.name,
        sample.index,
        ep.status,
        ep.reward,
        ep.n_turns,
        ep.prompt_len,
        ep.response_length,
        sum(ep.loss_mask),
        ep.generation_time_sec,
        ep.tool_time_sec,
        ep.tool_calls,
        ep.tool_result_truncations,
        ep.tito_prefix_checks,
        ep.early_compile_probes,
        ep.early_compile_successes,
        sample.non_generation_time,
        sample_wall_time_sec,
        ep.stop_detail,
    )
    return GenerateFnOutput(samples=sample)


generate.add_arguments = add_arguments
