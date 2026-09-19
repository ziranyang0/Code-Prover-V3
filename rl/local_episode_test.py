"""Local end-to-end test of the RL rollout path — no miles required.

Drives run_episode() with the LOCAL sglang server (native /generate, token
ids) and a local DockerSandbox on one training task, then prints trajectory
invariants and the verifier reward.

    PYTHONPATH=$PWD .venv/bin/python rl/local_episode_test.py \
        --task tasks/trainset_problems_300/<task_name> [--max-turns 40]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from rl.generate_with_prover import EpisodeConfig, run_episode
from rl.judge_config import add_judge_arguments, judge_options, preflight_judge
from rl.sandbox import DockerSandbox


def local_generate_fn(api_base: str, max_new_tokens: int, temperature: float):
    client = httpx.AsyncClient(timeout=600.0)

    async def generate_fn(input_ids: list[int], remaining_tokens: int) -> dict:
        resp = await client.post(f"{api_base}/generate", json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": min(max_new_tokens, remaining_tokens),
                "temperature": temperature,
                "no_stop_trim": True,
            },
            "return_logprob": True,
        })
        resp.raise_for_status()
        out = resp.json()
        pairs = out["meta_info"]["output_token_logprobs"]
        return {
            "token_ids": [p[1] for p in pairs],
            "logprobs": [p[0] for p in pairs],
            "finish_reason": out["meta_info"]["finish_reason"]["type"],
            "text": out["text"],
        }

    return generate_fn


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default="lizenan1995/code-prover-lean:latest")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--dump-transcript", default=None,
                    help="write the decoded token stream (full conversation) here")
    add_judge_arguments(ap)
    args = ap.parse_args()
    preflight_judge(args)

    task_dir = Path(args.task)
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
    cfg = EpisodeConfig(model_path=args.model_path, max_turns=args.max_turns,
                        docker_image=args.image, **judge_options(args))

    print(f"[test] starting sandbox from {args.image} ...")
    sandbox = await DockerSandbox.create(image=args.image)
    try:
        ep = await run_episode(
            local_generate_fn(args.api_base, args.max_new_tokens, args.temperature),
            sandbox, instruction, task_dir, cfg,
        )
    finally:
        await sandbox.close()

    n_trained = sum(ep.loss_mask)
    print(json.dumps({
        "status": ep.status,
        "stop_detail": ep.stop_detail,
        "n_turns": ep.n_turns,
        "prompt_len": ep.prompt_len,
        "total_tokens": len(ep.tokens),
        "response_length": ep.response_length,
        "trained_tokens": n_trained,
        "masked_tokens": ep.response_length - n_trained,
        "reward": ep.reward,
        "rewards": ep.rewards,
        "judge_details": ep.judge_details,
        "guard_events": ep.guard_events,
    }, indent=2, ensure_ascii=False))

    if args.dump_transcript:
        from rl.token_stream import TokenStream
        text = TokenStream(args.model_path).tokenizer.decode(ep.tokens)
        Path(args.dump_transcript).write_text(text, encoding="utf-8")
        print(f"[test] transcript ({len(ep.tokens)} tokens) -> {args.dump_transcript}")

    # invariants
    assert len(ep.loss_mask) == ep.response_length
    assert len(ep.logprobs) == ep.response_length
    assert 0 < n_trained <= ep.response_length
    print("[test] trajectory invariants OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
