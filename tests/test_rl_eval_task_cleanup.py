"""Exercise evaluation's actual task orchestration on failure and cancellation."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from miles.utils.types import Sample
from rl import evaluate


class EvaluationTaskCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'trials.jsonl'
        self.args = SimpleNamespace(
            prover_eval_concurrency=10, rollout_temperature=1., rollout_top_p=1.,
            rollout_top_k=-1, prover_max_total_tokens=100,
        )
        self.records = [{'prompt': 'fixture', 'metadata': {'task_name': 'task'}}]
        self.started, self.cleaned = set(), set()
        self.all_started = asyncio.Event()

    def imports(self, generate):
        return patch.dict(sys.modules, {
            'rl.generate_with_prover': SimpleNamespace(generate=generate),
            'miles.rollout.base_types': SimpleNamespace(GenerateFnInput=SimpleNamespace),
        })

    async def run_eval(self, count):
        return await evaluate.evaluate_records(
            self.args, self.records, budgets=[16], samples_per_prompt=count,
            output=self.output,
        )

    async def test_first_eight_failures_wait_for_remaining_trial_cleanup(self):
        async def generate(input):
            sample = input.sample
            self.started.add(sample.index)
            if len(self.started) == 10:
                self.all_started.set()
            await asyncio.wait_for(self.all_started.wait(), 2)
            if sample.index < 8:
                sample.status = Sample.Status.FAILED
                return SimpleNamespace(samples=sample)
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.005)
                self.cleaned.add(sample.index)

        with self.imports(generate):
            with self.assertRaisesRegex(RuntimeError, 'first eight evaluation episodes'):
                await self.run_eval(10)
        self.assertEqual(self.cleaned, {8, 9})
        self.assertEqual(len(self.output.read_text().splitlines()), 8)
        self.assertFalse(self.output.with_suffix('.summary.json').exists())

    async def test_completed_eval_preserves_comparator_audit_and_reward(self):
        audit = {"mode": "comparator", "artifact_dir": "/durable/proofs/fixture",
                 "comparator": {"accepted": True, "source_sha256": "fixture-hash"}}
        async def generate(input):
            sample = input.sample
            sample.status = Sample.Status.COMPLETED
            sample.reward = 1.0
            sample.metadata.update(judge_details=audit, rewards={"comparator_accepted": 1.0})
            return SimpleNamespace(samples=sample)

        with self.imports(generate):
            _, metrics = await self.run_eval(1)
        row = json.loads(self.output.read_text())
        self.assertEqual(row["metadata"]["judge_details"], audit)
        self.assertEqual(row["metadata"]["rewards"]["comparator_accepted"], 1.0)
        self.assertEqual(metrics["turn_16/pass@1"], 1.0)

    async def test_cancellation_waits_for_all_trial_cleanup(self):
        async def generate(input):
            self.started.add(input.sample.index)
            if len(self.started) == 2:
                self.all_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.005)
                self.cleaned.add(input.sample.index)

        with self.imports(generate):
            task = asyncio.create_task(self.run_eval(2))
            try:
                await asyncio.wait_for(self.all_started.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.cleaned, {0, 1})
        self.assertFalse(self.output.with_suffix('.summary.json').exists())


if __name__ == '__main__':
    unittest.main()
