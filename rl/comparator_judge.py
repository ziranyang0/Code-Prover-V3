"""Independent comparator judging for the existing rollout/evaluation loop."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex
import uuid

from verifier.comparator.backend import ComparatorInfrastructureError, MAX_SOURCE_BYTES, judge

_semaphore = None


async def grade_with_comparator(sandbox, tests_dir: Path, cfg, legacy_grade):
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(cfg.comparator_concurrency)
    if cfg.judge_mode not in ('shadow', 'comparator'):
        raise ValueError('expected shadow or comparator judge mode')
    queue_dir = getattr(cfg, "comparator_queue_dir", "")
    if (not cfg.comparator_image and not queue_dir) or not cfg.proof_artifacts_dir:
        raise ValueError('comparator requires an image or queue, and a durable proof artifacts directory')
    original = (tests_dir / 'original.lean').read_bytes().decode('utf-8')
    task_file = (tests_dir / 'task_file.txt').read_text().strip()
    # This read is not trusted execution: whatever it returns is the candidate
    # source, and must independently pass the clean comparator environment.
    capture = await sandbox.exec(
        f'head -c {MAX_SOURCE_BYTES + 1} -- {shlex.quote(task_file)}', timeout_sec=60)
    if capture.return_code or len(capture.stdout.encode()) > MAX_SOURCE_BYTES:
        raise ComparatorInfrastructureError('failed to capture final Lean source (or source too large)')
    artifact_dir = Path(cfg.proof_artifacts_dir).resolve() / uuid.uuid4().hex
    legacy = None
    # Save immediately, before any legacy verifier can execute candidate code.
    artifact_dir.mkdir(parents=True)
    (artifact_dir / 'solution.lean').write_bytes(capture.stdout.encode())
    (artifact_dir / 'original.lean').write_bytes(original.encode())
    if cfg.judge_mode == 'shadow':
        legacy = await legacy_grade(sandbox, tests_dir)
    try:
        if queue_dir:
            from verifier.comparator.queue import submit
            result = await submit(original, capture.stdout, queue=Path(queue_dir), timeout=cfg.comparator_timeout_sec)
        else:
            async with _semaphore:
                result = await judge(original, capture.stdout, image=cfg.comparator_image,
                                     artifacts=artifact_dir / 'comparator', timeout=cfg.comparator_timeout_sec)
        if type(result.get('accepted')) is not bool:
            raise ComparatorInfrastructureError('comparator returned no boolean verdict')
    except Exception as exc:
        result = {'status': 'infrastructure_error', 'error': str(exc), 'accepted': None}
        if cfg.judge_mode == 'comparator':
            details = {'mode': cfg.judge_mode, 'artifact_dir': str(artifact_dir), 'comparator': result}
            (artifact_dir / 'judging.json').write_text(json.dumps(details, indent=2, ensure_ascii=False))
            raise
    details = {'mode': cfg.judge_mode, 'artifact_dir': str(artifact_dir), 'comparator': result}
    if legacy is not None:
        reward, rewards = legacy
        rewards = dict(rewards)
        details['legacy_reward'] = reward
        details['disagreement'] = (None if result['accepted'] is None
                                   else bool(reward) != result['accepted'])
    else:
        reward = float(result['accepted'])
        rewards = {'reward': reward}
    # Unavailability is a separate metric, never a comparator proof rejection.
    rewards['comparator_available'] = float(result['accepted'] is not None)
    if result['accepted'] is not None:
        rewards['comparator_accepted'] = float(result['accepted'])
    (artifact_dir / 'judging.json').write_text(json.dumps(details, indent=2, ensure_ascii=False))
    return reward, rewards, details
