"""Default authoritative judging, deployment validation and audit persistence."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl.judge_config import add_judge_arguments, judge_options, preflight_judge
from verifier.comparator.backend import ComparatorInfrastructureError


class JudgeConfigurationTests(unittest.TestCase):
    def parse(self, *args):
        parser = argparse.ArgumentParser()
        with patch.dict(os.environ, {}, clear=True):
            add_judge_arguments(parser)
        return parser.parse_args(args)

    def test_comparator_default_requires_explicit_backend(self):
        from rl.generate_with_prover import EpisodeConfig, add_arguments
        self.assertEqual(EpisodeConfig(model_path="test").judge_mode, "comparator")
        parser = argparse.ArgumentParser()
        with patch.dict(os.environ, {}, clear=True):
            add_arguments(parser)
        args = parser.parse_args(["--prover-model-path", "test", "--prover-task-root", "/tasks"])
        self.assertEqual(args.prover_judge_mode, "comparator")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            judge_options(args)

    def test_explicit_legacy_requires_no_comparator(self):
        self.assertEqual(preflight_judge(self.parse("--prover-judge-mode", "legacy"))["judge_mode"], "legacy")

    def test_training_only_skips_backend_check(self):
        args = self.parse()
        args.training_only = True
        self.assertTrue(preflight_judge(args)["skipped"])

    def test_invalid_configuration_fails_early(self):
        valid = ["--prover-comparator-image", "test", "--prover-proof-artifacts-dir", "/durable/proofs"]
        for extra, error in [
            (["--prover-comparator-queue-dir", "/queue"], "exactly one"),
            (["--prover-proof-artifacts-dir", "relative"], "absolute"),
            (["--prover-proof-artifacts-dir", ""], "durable storage"),
            (["--prover-comparator-timeout-sec", "0"], "positive"),
            (["--prover-comparator-concurrency", "0"], "positive"),
        ]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, error):
                judge_options(self.parse(*valid, *extra))

    def test_preflight_requires_live_queue_and_writable_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.parse("--prover-comparator-queue-dir", folder,
                              "--prover-proof-artifacts-dir", str(root / "proofs"))
            with self.assertRaises(ComparatorInfrastructureError):
                preflight_judge(args)
            service = {"state": "ready", "heartbeat": time.time() - 60, "catalog_sha256": "test"}
            status = root / "service.json"
            status.write_text(json.dumps(service))
            with self.assertRaises(ComparatorInfrastructureError):
                preflight_judge(args)
            service["heartbeat"] = time.time()
            status.write_text(json.dumps(service))
            self.assertEqual(preflight_judge(args)["service"]["catalog_sha256"], "test")
            self.assertEqual(list((root / "proofs").iterdir()), [])
            args.prover_proof_artifacts_dir = str(status)
            with self.assertRaises(OSError):
                preflight_judge(args)

    def test_comparator_reserves_verification_time(self):
        from rl.generate_with_prover import effective_episode_timeout
        args = SimpleNamespace(prover_wall_time_budget_sec=2400, prover_router_timeout_sec=900,
                               prover_episode_timeout_sec=3300, prover_comparator_timeout_sec=1200)
        comparator = effective_episode_timeout(args)
        args.prover_judge_mode = "legacy"
        self.assertEqual(comparator - effective_episode_timeout(args), 1500)
