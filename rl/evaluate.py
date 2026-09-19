"""Evaluate fixed tasks through the same token loop, sandbox and verifier as RL."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

from rl.metrics import EpisodeMetrics
from rl.provenance import validate_prompt_data

logger = logging.getLogger(__name__)


def load_records(path, task_root):
    validate_prompt_data(Path(path), Path(task_root))
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def trial_plan(records, budgets, samples_per_prompt, seed):
    for group_index, record in enumerate(records):
        for sample_index in range(samples_per_prompt):
            sample_seed = int.from_bytes(hashlib.sha256(
                f"{seed}:{record['metadata']['task_name']}:{sample_index}".encode()
            ).digest()[:4], "big") & 0x7fffffff
            # Interleave budgets so time-dependent service load affects both arms.
            for budget in budgets:
                yield group_index, sample_index, sample_seed, budget, record


async def evaluate_records(args, records, *, budgets, samples_per_prompt, output, seed=20260907):
    from miles.rollout.base_types import GenerateFnInput
    from miles.utils.types import Sample
    from rl.generate_with_prover import generate

    if not records or samples_per_prompt < 1 or any(b < 1 for b in budgets):
        raise ValueError("evaluation needs records, positive budgets and samples per prompt")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to mix evaluation trials with {output}")
    semaphore = asyncio.Semaphore(args.prover_eval_concurrency)
    sampling = dict(temperature=args.rollout_temperature, top_p=args.rollout_top_p,
                    top_k=args.rollout_top_k, max_new_tokens=args.prover_max_total_tokens,
                    stop=getattr(args, "rollout_stop", None),
                    stop_token_ids=getattr(args, "rollout_stop_token_ids", None),
                    skip_special_tokens=False, no_stop_trim=True, spaces_between_special_tokens=False)
    results = []
    started = time.monotonic()

    async def one(plan, stream):
        gi, si, sample_seed, budget, record = plan
        async with semaphore:
            local_args = copy.copy(args)
            local_args.prover_max_tokens_per_turn = budget
            if routers := getattr(args, "prover_eval_routers", None):
                local_args.sglang_router_ip, local_args.sglang_router_port = routers[
                    (gi * samples_per_prompt + si) % len(routers)]
            sample = Sample(group_index=gi, index=gi * samples_per_prompt + si,
                            prompt=record["prompt"], metadata=copy.deepcopy(record["metadata"]))
            generated = await generate(GenerateFnInput(
                state=SimpleNamespace(args=local_args), sample=sample,
                sampling_params={**sampling, "sampling_seed": sample_seed}, evaluation=True))
            sample = generated.samples
            keys = ("stop_detail", "prompt_tokens", "model_generated_tokens", "tool_and_feedback_tokens",
                    "synthetic_tokens", "length_limited_turns", "n_turns", "tool_calls", "episode_wall_time_sec",
                    "judge_details", "rewards", "task_sha256", "tests_sha256")
            result = {"task_name": record["metadata"]["task_name"], "domain": record["metadata"].get("domain"),
                      "group_index": gi, "sample_index": si, "sampling_seed": sample_seed,
                      "turn_budget": budget, "reward": sample.reward, "status": sample.status.name,
                      "metadata": {key: sample.metadata[key] for key in keys if key in (sample.metadata or {})},
                      "weight_versions": sample.weight_versions}
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            results.append(result)
            if len(results) == 8 and all(r["status"] in {"FAILED", "ABORTED"} for r in results):
                raise RuntimeError("first eight evaluation episodes all failed infrastructure checks")
            if len(results) % 64 == 0:
                print(json.dumps({"completed_trials": len(results), "total_trials": len(records) * samples_per_prompt * len(budgets)}), flush=True)

    with output.open("x") as stream:
        tasks = [asyncio.create_task(one(plan, stream))
                 for plan in trial_plan(records, budgets, samples_per_prompt, seed)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    metrics = summarize_results(results, samples_per_prompt)
    metrics["wall_seconds"] = time.monotonic() - started
    output.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return results, metrics


def summarize_results(results, samples_per_prompt):
    groups = {}
    for row in results:
        groups.setdefault((row["turn_budget"], row["task_name"]), []).append(row)
    counters = {}
    for (budget, _), rows in groups.items():
        if len(rows) != samples_per_prompt or len({r["sample_index"] for r in rows}) != samples_per_prompt:
            raise ValueError("incomplete or duplicate evaluation group")
        samples = [SimpleNamespace(reward=r["reward"], status=SimpleNamespace(name=r["status"]),
                                   metadata=r["metadata"]) for r in rows]
        for scope in (f"turn_{budget}", f"turn_{budget}/{rows[0]['domain']}"):
            counters.setdefault(scope, EpisodeMetrics()).add_group(samples)
    return {key: value for scope, counter in counters.items() for key, value in counter.summarize(scope).items()}


def generate_rollout_eval(args, rollout_id, data_source, evaluation=False):
    from miles.rollout.base_types import RolloutFnEvalOutput
    from rl.fully_async_rollout import get_global_worker, stop_global_worker

    try:
        if not evaluation or not args.prover_eval_prompt_data:
            raise ValueError("fixed evaluation requires --prover-eval-prompt-data")
        final_eval = rollout_id + 1 >= args.num_rollout
        samples_per_prompt = (args.prover_full_eval_samples_per_prompt if final_eval
                              else args.n_samples_per_eval_prompt)
        records = load_records(args.prover_eval_prompt_data, args.prover_task_root)
        # Use the sampler's loop: its HTTP connections and sandbox semaphore are
        # bound there. A separate asyncio.run/run loop cannot safely reuse them.
        worker = get_global_worker(args, data_source)
        worker.pause(timeout_sec=args.prover_async_pause_timeout_sec)
        local_args = copy.copy(args)
        local_args.prover_eval_concurrency = min(
            args.prover_sandbox_concurrency,
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine,
        )
        output = Path(args.save) / "eval" / f"rollout_{rollout_id}.jsonl"
        coroutine = evaluate_records(local_args, records, budgets=[args.prover_max_tokens_per_turn],
                                     samples_per_prompt=samples_per_prompt, output=output)
        _, metrics = asyncio.run_coroutine_threadsafe(coroutine, worker.event_loop).result()
        result = RolloutFnEvalOutput(data={}, metrics={f"eval/d3_hard/{key}": value for key, value in metrics.items()})
    except BaseException:
        # Miles aborts training when eval raises. Never resume admission on
        # this path, including failures while loading records or pausing.
        logger.exception("rollout %s evaluation failed; stopping sampler and reclaiming sandboxes", rollout_id)
        try:
            stop_global_worker()
        except Exception:
            logger.exception("evaluation cleanup also failed; preserving original error")
        raise
    else:
        if final_eval:
            worker.stop()
        else:
            worker.resume()
        return result


def main():
    from miles.utils import http_utils
    from rl.generate_with_prover import add_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--router-host", required=True)
    parser.add_argument("--router-port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--turn-budgets", type=int, nargs="+", default=[2048])
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--prover-eval-concurrency", type=int, default=128)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=-1)
    args = parser.parse_args()
    args.sglang_router_ip, args.sglang_router_port = args.router_host, args.router_port
    args.rollout_num_gpus = args.rollout_num_gpus_per_engine = 1
    args.sglang_server_concurrency = args.prover_eval_concurrency
    args.use_distributed_post = False
    records = load_records(args.prompt_data, args.prover_task_root)

    async def run_all():
        from rl.warm_pool import start_warm_pool

        http_utils.init_http_client(args)
        pool = None
        try:
            pool = await start_warm_pool(args)
            _, summary = await evaluate_records(args, records, budgets=args.turn_budgets,
                                                samples_per_prompt=args.samples_per_prompt, output=args.out)
            print(json.dumps(summary, indent=2))
        finally:
            if pool is not None:
                await pool.close()
            if http_utils._http_client is not None:
                await http_utils._http_client.aclose()
                http_utils._http_client = None
    asyncio.run(run_all())


if __name__ == "__main__":
    main()
