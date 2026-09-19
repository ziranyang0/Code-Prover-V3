"""Fault injection for bounded rollout recovery and per-episode diagnostics."""
import asyncio
import importlib.util
import os
import pickle
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from rl import diagnostics, sandbox as backend
from rl.errors import safe_failure_detail


class Sample:
    class Status:
        COMPLETED = "COMPLETED"
        TRUNCATED = "TRUNCATED"
        FAILED = "FAILED"
        ABORTED = "ABORTED"

    def __init__(self, index=0, failed=False):
        self.index = index
        self.status = self.Status.FAILED if failed else "COMPLETED"
        self.metadata = {"weight_versions_complete": True}
        self.oldest_weight_version = None
        self.reward = 0.0  # A correctly graded wrong answer is healthy.
        self.prompt = self.response = self.label = ""

    def reset_for_retry(self):
        self.status = self.Status.ABORTED


def async_module():
    modules = {name: ModuleType(name) for name in (
        "miles", "miles.rollout", "miles.rollout.sglang_rollout", "miles.rollout.base_types",
        "miles.utils", "miles.utils.async_utils", "miles.utils.types",
    )}
    modules["miles.rollout.sglang_rollout"].GenerateState = object
    modules["miles.rollout.sglang_rollout"].generate_and_rm_group = lambda *_: None
    modules["miles.rollout.base_types"].RolloutFnTrainOutput = SimpleNamespace
    modules["miles.utils.async_utils"].run = lambda coro: coro
    modules["miles.utils.types"].Sample = Sample
    path = Path(__file__).parents[1] / "rl/fully_async_rollout.py"
    spec = importlib.util.spec_from_file_location("_reliability_async", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class GroupRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = async_module()
        self.args = SimpleNamespace(
            rollout_global_dataset=True, rollout_batch_size=1, n_samples_per_prompt=1,
            dynamic_sampling_filter_path=None, max_weight_staleness=None,
            prover_async_max_group_retries=2, prover_async_no_progress_timeout_sec=100,
            num_rollout=100,
        )
        self.source = SimpleNamespace(
            retry_completed=lambda *_: None, discard_completed=lambda *_: None,
            record_accepted=lambda *_: None,
        )

    async def collect(self, pop):
        worker = SimpleNamespace(raise_if_failed=lambda: None, pop_completed=pop)
        with patch.object(self.module, "get_global_worker", return_value=worker):
            return await self.module.generate_rollout_async(self.args, 0, self.source)

    async def test_retry_preserves_healthy_group_members(self):
        self.args.n_samples_per_prompt = 3
        healthy, truncated, failed = Sample(1), Sample(2), Sample(3, failed=True)
        truncated.status = Sample.Status.TRUNCATED
        healthy.response = "keep proof"
        healthy.oldest_weight_version = 7
        group = [healthy, truncated, failed]
        completed = deque([(1, group)])

        def retry(_group_id, samples):
            self.assertIs(samples[0], healthy)
            self.assertEqual(healthy.status, Sample.Status.COMPLETED)
            self.assertEqual(healthy.response, "keep proof")
            self.assertEqual(healthy.oldest_weight_version, 7)
            self.assertEqual(truncated.status, Sample.Status.TRUNCATED)
            self.assertEqual(failed.status, Sample.Status.ABORTED)
            restored = pickle.loads(pickle.dumps(samples))
            restored[2].status = Sample.Status.COMPLETED
            completed.append((2, restored))

        self.source.retry_completed = retry
        result = await self.collect(completed.popleft)
        self.assertEqual(result.samples[0][0].response, "keep proof")
        self.assertEqual(result.metrics["sampling/infrastructure_retried_groups"], 1)

    async def test_pause_preserves_completed_members(self):
        healthy, pending = Sample(1), Sample(2, failed=True)
        worker = self.module.AsyncRolloutWorker.__new__(self.module.AsyncRolloutWorker)
        recycled = []
        worker.data_source = SimpleNamespace(retry_inflight=recycled.append)
        worker._recycle_inflight([healthy, pending])
        self.assertEqual(healthy.status, Sample.Status.COMPLETED)
        self.assertEqual(pending.status, Sample.Status.ABORTED)
        self.assertEqual(recycled, [[healthy, pending]])

    async def test_preserved_members_still_obey_weight_staleness(self):
        self.args.max_weight_staleness = 2
        old = Sample(1)
        old.oldest_weight_version = 1
        old.metadata["weight_version_origin"] = 0
        completed = deque([(1, [old])])

        def retry(_group_id, group):
            self.assertEqual(group[0].status, Sample.Status.ABORTED)
            fresh = Sample(2)
            fresh.oldest_weight_version = 5
            fresh.metadata["weight_version_origin"] = 0
            completed.append((2, [fresh]))

        self.source.retry_completed = retry
        with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(return_value=5)):
            result = await self.collect(completed.popleft)
        self.assertEqual(result.samples[0][0].index, 2)
        self.assertEqual(result.metrics["sampling/stale_groups"], 1)

    async def test_verification_budget_extends_episode_and_progress_deadlines(self):
        self.args.prover_wall_time_budget_sec = 2400
        self.args.prover_router_timeout_sec = 900
        self.args.prover_episode_timeout_sec = 3300
        self.args.prover_async_no_progress_timeout_sec = 3600
        healthy = Sample(9)
        clock = SimpleNamespace(monotonic=iter([0, 4900, 4900]).__next__)
        with patch.object(self.module, "time", clock):
            result = await self.collect(lambda: (9, [healthy]))
        self.assertEqual(result.samples, [[healthy]])
        self.assertEqual(self.args.prover_episode_timeout_sec, 6670)
        self.assertEqual(self.args.prover_async_no_progress_timeout_sec, 6970)

    async def test_invalid_progress_timeout_is_not_hidden_by_budget_expansion(self):
        self.args.prover_wall_time_budget_sec = 2400
        self.args.prover_router_timeout_sec = 900
        self.args.prover_episode_timeout_sec = 3300
        self.args.prover_async_no_progress_timeout_sec = -1
        with self.assertRaisesRegex(ValueError, "no-progress timeout"):
            await self.collect(lambda: (1, [Sample()]))

    async def test_restart_counter_cannot_accept_old_or_unprovenanced_samples(self):
        self.args.max_weight_staleness = 2
        self.args.start_rollout_id = 40
        self.args.update_weights_interval = 1
        for provenance in ({"weight_versions_complete": True, "weight_version_origin": 0}, {}):
            old = Sample(7)
            old.oldest_weight_version = 7
            old.metadata = provenance
            fresh = Sample(8)
            fresh.oldest_weight_version = 1
            fresh.metadata = {"weight_versions_complete": True, "weight_version_origin": 40}
            completed = deque([(1, [old])])
            self.source.retry_completed = lambda _gid, _group: completed.append((2, [fresh]))
            with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(return_value=1)):
                result = await self.collect(completed.popleft)
            self.assertIs(result.samples[0][0], fresh)
            self.assertEqual(result.metrics["sampling/stale_groups"], 1)

    async def test_recent_previous_run_sample_is_still_reusable(self):
        self.args.max_weight_staleness = 2
        self.args.start_rollout_id = 40
        self.args.update_weights_interval = 1
        old = Sample(7)
        old.oldest_weight_version = 6
        old.metadata = {"weight_versions_complete": True, "weight_version_origin": 35}
        with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(return_value=1)):
            result = await self.collect(lambda: (1, [old]))
        self.assertIs(result.samples[0][0], old)
        self.assertEqual(result.metrics["sampling/stale_groups"], 0)

    async def test_unavailable_engine_version_does_not_accept_samples(self):
        self.args.max_weight_staleness = 2
        self.args.prover_async_no_progress_timeout_sec = .01
        with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(return_value=None)):
            with self.assertRaisesRegex(TimeoutError, "engine weight version unavailable"):
                await self.collect(lambda: (1, [Sample()]))

    async def test_transient_version_query_recovers_without_discarding_group(self):
        self.args.max_weight_staleness = 2
        sample = Sample(1)
        sample.oldest_weight_version = 1
        sample.metadata = {"weight_versions_complete": True, "weight_version_origin": 0}
        with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(side_effect=[None, 1, 1])):
            result = await self.collect(lambda: (1, [sample]))
        self.assertIs(result.samples[0][0], sample)

    async def test_changed_weight_update_interval_requires_regeneration(self):
        sample = Sample(1)
        sample.oldest_weight_version = 4
        sample.metadata = {"weight_versions_complete": True, "weight_version_origin": 3, "weight_update_interval": 2,
                           "weight_version_start_rollout_id": 6}
        self.assertIsNone(self.module._group_weight_version([sample], update_interval=1))
        self.assertEqual(self.module._group_weight_version([sample], update_interval=2, start_rollout_id=6), 7)

    async def test_unaligned_restart_rejects_colliding_and_legacy_versions(self):
        self.args.start_rollout_id = 9
        self.args.update_weights_interval = 10
        self.args.max_weight_staleness = 0
        for old_start in (0, None):
            with self.subTest(old_start=old_start):
                old, fresh = Sample(7), Sample(8)
                old.oldest_weight_version = fresh.oldest_weight_version = 1
                old.metadata = {"weight_versions_complete": True, "weight_version_origin": 0, "weight_update_interval": 10}
                if old_start is not None:
                    old.metadata["weight_version_start_rollout_id"] = old_start
                fresh.metadata = {"weight_versions_complete": True, "weight_version_origin": 0, "weight_update_interval": 10,
                                  "weight_version_start_rollout_id": 9}
                completed = deque([(1, [old])])
                retries = []
                def retry(gid, group):
                    retries.append(gid)
                    completed.append((2, [fresh]))
                self.source.retry_completed = retry
                with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(return_value=1)):
                    result = await self.collect(completed.popleft)
                self.assertEqual(retries, [1])
                self.assertIs(result.samples[0][0], fresh)
                self.assertEqual(result.metrics["sampling/stale_groups"], 1)

    async def test_same_start_still_checks_periodic_sample_age(self):
        self.args.start_rollout_id = 9
        self.args.update_weights_interval = 10
        self.args.max_weight_staleness = 0
        old, fresh = Sample(7), Sample(8)
        old.oldest_weight_version, fresh.oldest_weight_version = 1, 2
        for sample in (old, fresh):
            sample.metadata = {"weight_versions_complete": True, "weight_version_origin": 0, "weight_update_interval": 10,
                               "weight_version_start_rollout_id": 9}
        completed = deque([(1, [old])])
        self.source.retry_completed = lambda *_: completed.append((2, [fresh]))
        with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(return_value=2)):
            result = await self.collect(completed.popleft)
        self.assertIs(result.samples[0][0], fresh)
        self.assertEqual(result.metrics["sampling/stale_groups"], 1)

    async def test_incomplete_turn_versions_and_legacy_samples_are_retried(self):
        from rl.generate_with_prover import weight_versions_complete

        self.args.start_rollout_id = 40
        self.args.update_weights_interval = 1
        self.args.max_weight_staleness = 0
        for versions in (["1", "None"], ["1", "unknown"], ["1", "0"], None):
            with self.subTest(versions=versions):
                old, fresh = Sample(7), Sample(8)
                old.oldest_weight_version = fresh.oldest_weight_version = 1
                old.metadata = {"weight_version_origin": 40}
                if versions is not None:
                    old.metadata["weight_versions_complete"] = weight_versions_complete(versions)
                fresh.metadata = {"weight_version_origin": 40,
                                  "weight_versions_complete": weight_versions_complete(["1", "1"])}
                completed = deque([(1, [old])])
                self.source.retry_completed = lambda *_: completed.append((2, [fresh]))
                with patch.object(self.module._cached_weight_version, "get", new=AsyncMock(return_value=1)):
                    result = await self.collect(completed.popleft)
                self.assertIs(result.samples[0][0], fresh)
                self.assertEqual(result.metrics["sampling/stale_groups"], 1)

    def test_version_completeness_requires_positive_integer_for_every_request(self):
        from rl.generate_with_prover import weight_versions_complete

        for versions in ([], ["1", "None"], ["1", "unknown"], ["0"], ["-1"],
                         ["1.5"], ["True"], ["²"], ["１"]):
            with self.subTest(versions=versions):
                self.assertFalse(weight_versions_complete(versions))
        self.assertTrue(weight_versions_complete(["1", "2"]))

    async def test_expired_cache_refresh_failure_waits_for_a_real_version(self):
        self.args.max_weight_staleness = 2
        self.args.sglang_router_ip, self.args.sglang_router_port = "unused", 1
        cache = self.module._CachedWeightVersion(ttl=1)
        cache.value, cache.last_query = 1, 0
        response = AsyncMock()
        response.status = 200
        response.json.return_value = {"weight_version": "5"}
        request = AsyncMock()
        request.__aenter__.return_value = response
        session = AsyncMock()
        session.get = MagicMock(return_value=request)
        context = AsyncMock()
        context.__aenter__.return_value = session
        old, fresh = Sample(1), Sample(2)
        old.oldest_weight_version, fresh.oldest_weight_version = 1, 5
        old.metadata = fresh.metadata = {"weight_versions_complete": True, "weight_version_origin": 0}
        completed = deque([(1, [old])])
        self.source.retry_completed = lambda _gid, _group: completed.append((2, [fresh]))
        with patch.object(self.module, "_cached_weight_version", cache), \
             patch.object(self.module.aiohttp, "ClientSession", side_effect=[OSError("offline"), context]) as client:
            result = await self.collect(completed.popleft)
        self.assertIs(result.samples[0][0], fresh)
        self.assertEqual(result.metrics["sampling/stale_groups"], 1)
        self.assertEqual(client.call_count, 2)

    async def test_non_success_or_invalid_response_cannot_reuse_expired_cache(self):
        args = SimpleNamespace(sglang_router_ip="unused", sglang_router_port=1)
        for status, payload in [(503, {}), (200, {}), (200, {"weight_version": "default"}),
                                (200, {"weight_version": "0"})]:
            with self.subTest(status=status, payload=payload):
                cache = self.module._CachedWeightVersion(ttl=0)
                cache.value = 7
                response = AsyncMock()
                response.status, response.json.return_value = status, payload
                request = AsyncMock()
                request.__aenter__.return_value = response
                session = AsyncMock()
                session.get = MagicMock(return_value=request)
                context = AsyncMock()
                context.__aenter__.return_value = session
                with patch.object(self.module.aiohttp, "ClientSession", return_value=context):
                    self.assertIsNone(await cache.get(args))
                    self.assertIsNone(cache.value)

    async def test_three_failed_groups_do_not_stop_training(self):
        completed = deque((i, [Sample(i, failed=True)]) for i in range(3))
        healthy = Sample(9)
        completed.append((9, [healthy]))
        result = await self.collect(completed.popleft)
        self.assertEqual(result.samples, [[healthy]])
        self.assertEqual(result.metrics["sampling/infrastructure_retried_groups"], 3)

    async def test_retry_budget_survives_requeue_and_serialization(self):
        completed = deque([(10, [Sample(failed=True)])])
        retried, discarded = [], []
        def retry(group_id, group):
            retried.append(group_id)
            # Simulate a checkpoint roundtrip and a different worker group ID.
            restored = pickle.loads(pickle.dumps(group))
            completed.append((group_id + 1, restored))
        def discard(group_id):
            discarded.append(group_id)
            completed.append((20, [Sample(20)]))
        self.source.retry_completed = retry
        self.source.discard_completed = discard
        result = await self.collect(completed.popleft)
        self.assertEqual(retried, [10, 11])
        self.assertEqual(discarded, [12])
        self.assertEqual(result.metrics["sampling/infrastructure_exhausted_groups"], 1)

    async def test_no_progress_stops_even_when_failed_queue_never_empties(self):
        self.args.prover_async_max_group_retries = 0
        self.args.prover_async_no_progress_timeout_sec = 3
        clock = SimpleNamespace(monotonic=iter([0, 1, 2, 4]).__next__)
        with patch.object(self.module, "time", clock):
            with self.assertRaisesRegex(TimeoutError, "no accepted.*exhausted=2"):
                await self.collect(lambda: (1, [Sample(failed=True)]))

    async def test_no_progress_stops_empty_queue(self):
        clock = SimpleNamespace(monotonic=iter([0, 101]).__next__)
        with patch.object(self.module, "time", clock):
            with self.assertRaisesRegex(TimeoutError, "no accepted"):
                await self.collect(lambda: None)

    async def test_default_allows_long_episode_past_old_2700s_deadline(self):
        from argparse import ArgumentParser
        parser = ArgumentParser()
        self.module.add_arguments(parser)
        self.args.prover_async_no_progress_timeout_sec = parser.parse_args([]).prover_async_no_progress_timeout_sec
        self.args.prover_episode_timeout_sec = 3300
        healthy = Sample(9)
        clock = SimpleNamespace(monotonic=iter([0, 3000, 3000]).__next__)
        with patch.object(self.module, "time", clock):
            result = await self.collect(lambda: (9, [healthy]))
        self.assertEqual(result.samples, [[healthy]])

    async def test_long_episode_default_still_stops_after_3600s(self):
        self.args.prover_async_no_progress_timeout_sec = None
        self.args.prover_episode_timeout_sec = 3300
        clock = SimpleNamespace(monotonic=iter([0, 3601]).__next__)
        with patch.object(self.module, "time", clock):
            with self.assertRaisesRegex(TimeoutError, "no accepted.*3600s"):
                await self.collect(lambda: None)

    async def test_worker_crash_is_still_fatal(self):
        def crash():
            raise RuntimeError("worker crashed")
        worker = SimpleNamespace(raise_if_failed=crash)
        with patch.object(self.module, "get_global_worker", return_value=worker):
            with self.assertRaisesRegex(RuntimeError, "worker crashed"):
                await self.module.generate_rollout_async(self.args, 0, self.source)


class SafeUploadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "proof"
        self.source.write_bytes(b"theorem unchanged")
        self.raw = SimpleNamespace(
            sandbox_id="sandbox-test", kill=AsyncMock(),
            commands=SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_code=0, stderr="", stdout=""))),
            files=SimpleNamespace(write=AsyncMock()),
        )
        self.sandbox = backend.E2BSandbox(self.raw)
        self.sandbox._reset_envd_connection = AsyncMock()

    async def upload(self):
        with patch.object(backend.asyncio, "sleep", AsyncMock()):
            await self.sandbox.upload_file(self.source, "/task/proof.lean")

    async def test_write_retry_reconnects_and_preserves_exact_bytes(self):
        self.raw.files.write.side_effect = [httpx.WriteError("closed"), None]
        await self.upload()
        self.sandbox._reset_envd_connection.assert_awaited_once()
        self.assertEqual(self.raw.commands.run.await_count, 1)
        self.assertEqual(self.raw.files.write.await_count, 2)
        self.assertEqual(self.raw.files.write.await_args_list[0], self.raw.files.write.await_args_list[1])
        self.raw.kill.assert_not_awaited()

    async def test_mkdir_retry_is_bounded_and_never_uploads_on_failure(self):
        self.raw.commands.run.side_effect = httpx.WriteError("closed")
        with self.assertRaises(httpx.WriteError):
            await self.upload()
        self.assertEqual(self.raw.commands.run.await_count, 3)
        self.assertEqual(self.sandbox._reset_envd_connection.await_count, 2)
        self.raw.files.write.assert_not_awaited()

    async def test_permission_failure_is_not_retried(self):
        self.raw.commands.run.return_value.exit_code = 1
        self.raw.commands.run.return_value.stderr = "permission denied"
        with self.assertRaisesRegex(RuntimeError, "permission denied"):
            await self.upload()
        self.raw.commands.run.assert_awaited_once()
        self.sandbox._reset_envd_connection.assert_not_awaited()

    async def test_auth_failure_is_not_retried(self):
        request = httpx.Request("POST", "https://example.invalid")
        self.raw.files.write.side_effect = httpx.HTTPStatusError(
            "unauthorized", request=request, response=httpx.Response(401, request=request),
        )
        with self.assertRaises(httpx.HTTPStatusError):
            await self.upload()
        self.sandbox._reset_envd_connection.assert_not_awaited()

    async def test_arbitrary_command_is_never_replayed(self):
        self.raw.commands.run.side_effect = httpx.WriteError("closed")
        result = await self.sandbox.exec("arbitrary-command")
        self.assertEqual(result.return_code, 1)
        self.raw.commands.run.assert_awaited_once()
        self.sandbox._reset_envd_connection.assert_not_awaited()

    async def test_cancellation_is_not_retried(self):
        self.raw.files.write.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.upload()
        self.sandbox._reset_envd_connection.assert_not_awaited()

    async def test_operation_deadline_is_not_mislabeled_episode_timeout(self):
        async def hang(*_):
            await asyncio.Event().wait()
        self.raw.files.write.side_effect = hang
        with patch.object(backend, "SAFE_OPERATION_TIMEOUT_SEC", .01):
            with self.assertRaisesRegex(backend.SandboxOperationTimeout, "envd.files.write"):
                await self.sandbox.upload_file(self.source, "/task/proof.lean")

    async def test_reconnect_closes_only_owned_client(self):
        raw = SimpleNamespace(_envd_api=SimpleNamespace(aclose=AsyncMock()))
        sandbox = backend.E2BSandbox(raw)
        with self.assertRaisesRegex(RuntimeError, "unowned"):
            await sandbox._reset_envd_connection()
        raw._envd_api.aclose.assert_not_awaited()
        setattr(raw, backend._OWNS_ENVD_TRANSPORT_ATTR, True)
        with patch.object(backend, "_configure_envd_client") as wire:
            await sandbox._reset_envd_connection()
        raw._envd_api.aclose.assert_awaited_once()
        wire.assert_called_once_with(raw)


    async def test_pinned_sdk_reconnect_rewires_all_surfaces(self):
        try:
            from e2b import AsyncSandbox
            from e2b.connection_config import ConnectionConfig
            from packaging.version import Version
        except ImportError:
            self.skipTest("requires the pinned E2B SDK")
        cls = backend._isolated_async_sandbox_class(AsyncSandbox)
        raw = cls(sandbox_id="local-test", envd_version=Version("0.5.0"),
                  envd_access_token=None, sandbox_domain="example.invalid",
                  connection_config=ConnectionConfig())
        setattr(raw, backend._OWNS_ENVD_TRANSPORT_ATTR, True)
        old_client, old_transport = raw._envd_api, raw._transport
        old_commands, old_files, old_pty, old_git = raw.commands, raw.files, raw.pty, raw.git
        try:
            await backend.E2BSandbox(raw)._reset_envd_connection()
            self.assertTrue(old_client.is_closed)
            self.assertIsNot(raw._transport, old_transport)
            self.assertIsNot(raw.commands, old_commands)
            self.assertIsNot(raw.files, old_files)
            self.assertIsNot(raw.pty, old_pty)
            self.assertIsNot(raw.git, old_git)
            self.assertIs(raw.files._envd_api, raw._envd_api)
            self.assertIs(raw.files._pool, raw._transport.pool)
            self.assertEqual(raw.sandbox_id, "local-test")
        finally:
            await raw._envd_api.aclose()


class GenerationBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def episode(self, *, hang_tool=False, slow_grade=False, inner_timeout=False, version=None):
        from rl import generate_with_prover as prover

        class Stream:
            def __init__(self, _model):
                self.tokenizer = SimpleNamespace(decode=lambda *_a, **_k: "answer")
                self.glue = SimpleNamespace(im_end_id=99)
                self.tokens, self.loss_mask, self.logprobs = [], [], []
                self.prompt_len = 0
            def start(self, _instruction):
                self.tokens = [1, 2]
                self.prompt_len = 2
            def append_generated(self, ids, probs):
                self.tokens.extend(ids)
                self.loss_mask.extend([1] * len(ids))
                self.logprobs.extend(probs)
            def validate(self):
                pass

        async def tool(*_args):
            await asyncio.Event().wait()
        surface = SimpleNamespace(setup=AsyncMock(), _read_task_file=AsyncMock(return_value="sorry"),
                                  _dispatch=tool)
        async def generate(*_args):
            if inner_timeout:
                raise TimeoutError("request deadline")
            return dict(token_ids=[7, 99], logprobs=[-.1, -.2], finish_reason="stop",
                        duration_sec=0., weight_version=version)
        async def grade(*_args):
            if slow_grade:
                await asyncio.sleep(.08)
            return 1., {"reward": 1.}
        grader = AsyncMock(side_effect=grade)
        decoded = SimpleNamespace(errors=[], tool_calls=(
            [SimpleNamespace(name="Bash", arguments={"timeout": 3600000})] if hang_tool else []))
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / "tests").mkdir()
            (task / "tests/task_file.txt").write_text("/task/test.lean")
            with patch.object(prover, "TokenStream", Stream), \
                 patch.object(prover, "_ToolSurface", return_value=surface), \
                 patch.object(prover.codec, "decode_assistant", return_value=decoded), \
                 patch.object(prover, "_grade", grader):
                try:
                    return await prover.run_episode(generate, SimpleNamespace(upload_file=AsyncMock()),
                        "proof", task, prover.EpisodeConfig(judge_mode="legacy", model_path="fake", max_turns=1,
                        max_total_tokens=32, wall_time_budget_sec=.02, router_timeout_sec=.03))
                finally:
                    if hang_tool or inner_timeout:
                        grader.assert_not_awaited()

    async def test_episode_keeps_missing_and_invalid_version_entries(self):
        from rl.generate_with_prover import weight_versions_complete
        for version, complete in ((None, False), ("unknown", False), ("1", True)):
            with self.subTest(version=version):
                result = await self.episode(version=version)
                self.assertEqual(result.weight_versions, [str(version)])
                self.assertEqual(weight_versions_complete(result.weight_versions), complete)

    async def test_tool_cannot_consume_verification_reserve(self):
        from rl.generate_with_prover import GenerationTimeout
        with self.assertRaises(GenerationTimeout):
            await asyncio.wait_for(self.episode(hang_tool=True), timeout=.5)

    async def test_verification_runs_beyond_generation_deadline_after_clean_finish(self):
        result = await asyncio.wait_for(self.episode(slow_grade=True), timeout=.5)
        self.assertEqual(result.reward, 1.)

    async def test_inner_timeout_is_not_mislabeled_as_generation_budget(self):
        with self.assertRaisesRegex(TimeoutError, "request deadline"):
            await self.episode(inner_timeout=True)


class VerificationPhaseBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_has_an_aggregate_deadline_and_never_starts_verification_on_timeout(self):
        from rl import generate_with_prover as prover
        async def upload(*_args):
            await asyncio.sleep(.04)
        sandbox = SimpleNamespace(upload_file=AsyncMock(side_effect=upload), exec=AsyncMock())
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / "a").write_text("a")
            (task / "b").write_text("b")
            with patch.object(prover, "GRADE_UPLOAD_TIMEOUT_SEC", .06):
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(prover._grade(sandbox, task), timeout=.5)
        sandbox.exec.assert_not_awaited()

    async def test_slow_successful_phases_have_separate_budgets(self):
        from rl import generate_with_prover as prover
        from rl.sandbox import ExecResult
        async def upload(*_args):
            await asyncio.sleep(.03)
        async def execute(command, **_kwargs):
            if command == "bash /tests/test.sh":
                await asyncio.sleep(.08)
                return ExecResult("", "", 0)
            await asyncio.sleep(.03)
            return ExecResult('{"reward": 1}', "", 0)
        sandbox = SimpleNamespace(upload_file=upload, exec=execute)
        args = SimpleNamespace(prover_wall_time_budget_sec=.01, prover_router_timeout_sec=.01,
                               prover_episode_timeout_sec=.01)
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            (task / "test.sh").write_text("true")
            with patch.object(prover, "GRADE_UPLOAD_TIMEOUT_SEC", .1), \
                 patch.object(prover, "GRADE_TIMEOUT_SEC", .12), \
                 patch.object(prover, "GRADE_REWARD_TIMEOUT_SEC", .08), \
                 patch.object(prover, "EPISODE_SCHEDULING_MARGIN_SEC", .1):
                remaining = prover.effective_episode_timeout(args) - .02
                async with asyncio.timeout(remaining):
                    reward, _ = await prover._grade(sandbox, task)
        self.assertEqual(reward, 1.)

    async def test_execution_and_reward_reads_each_stop_at_their_own_deadline(self):
        from rl import generate_with_prover as prover
        from rl.sandbox import ExecResult
        for hung_phase in ("exec", "reward"):
            with self.subTest(phase=hung_phase):
                async def execute(command, **_kwargs):
                    if hung_phase == "exec" or command.startswith("cat "):
                        await asyncio.Event().wait()
                    return ExecResult("", "", 0)
                sandbox = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(side_effect=execute))
                with tempfile.TemporaryDirectory() as directory, \
                     patch.object(prover, "GRADE_TIMEOUT_SEC", .03), \
                     patch.object(prover, "GRADE_REWARD_TIMEOUT_SEC", .03):
                    with self.assertRaises(TimeoutError):
                        await asyncio.wait_for(prover._grade(sandbox, Path(directory)), timeout=.5)
                self.assertEqual(sandbox.exec.await_count, 1 if hung_phase == "exec" else 2)


class VerificationBudgetTests(unittest.TestCase):
    def test_current_campaign_budget_covers_verifier_and_last_request(self):
        from rl.generate_with_prover import effective_episode_timeout, GRADE_TIMEOUT_SEC
        args = SimpleNamespace(prover_judge_mode="legacy", prover_wall_time_budget_sec=2400,
                               prover_router_timeout_sec=900, prover_episode_timeout_sec=3300)
        self.assertEqual(GRADE_TIMEOUT_SEC, 1320)
        self.assertEqual(effective_episode_timeout(args), 5170)
        args.prover_episode_timeout_sec = 6000
        self.assertEqual(effective_episode_timeout(args), 6000)


class RewardReadRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_read_failure_does_not_repeat_verification(self):
        from rl.generate_with_prover import _grade
        from rl.sandbox import ExecResult
        sandbox = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(side_effect=[
            ExecResult("", "", 0), httpx.WriteError("closed"),
            ExecResult('{"reward": 0}', "", 0),
        ]))
        with tempfile.TemporaryDirectory() as directory, patch("rl.generate_with_prover.asyncio.sleep", new_callable=AsyncMock):
            reward, _ = await _grade(sandbox, Path(directory))
        self.assertEqual(reward, 0)
        commands = [call.args[0] for call in sandbox.exec.call_args_list]
        self.assertEqual(commands.count("bash /tests/test.sh"), 1)
        self.assertEqual(sandbox.exec.call_args_list[0].kwargs["timeout_sec"], 1320)
        self.assertEqual(commands.count("cat /logs/verifier/reward.json"), 2)

    async def test_read_retries_are_bounded_and_remain_infrastructure_failures(self):
        from rl.generate_with_prover import _grade
        from rl.sandbox import ExecResult
        sandbox = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(side_effect=[
            ExecResult("", "", 0), *[httpx.WriteError("closed") for _ in range(3)],
        ]))
        with tempfile.TemporaryDirectory() as directory, patch("rl.generate_with_prover.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                await _grade(sandbox, Path(directory))
        self.assertEqual(sandbox.exec.await_count, 4)

    async def test_cancellation_is_not_retried(self):
        from rl.generate_with_prover import _grade
        from rl.sandbox import ExecResult
        sandbox = SimpleNamespace(upload_file=AsyncMock(), exec=AsyncMock(side_effect=[
            ExecResult("", "", 0), asyncio.CancelledError(),
        ]))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(asyncio.CancelledError):
                await _grade(sandbox, Path(directory))
        self.assertEqual(sandbox.exec.await_count, 2)


class DiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_episode_context_survives_child_timeout(self):
        async def episode(index):
            state = diagnostics.EpisodeDiagnostics("task", index, sandbox_id=f"sandbox-{index}")
            token = diagnostics.current_episode.set(state)
            async def child():
                diagnostics.set_phase("verification")
                with diagnostics.operation("verifier.exec"):
                    await asyncio.Event().wait()
            try:
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(child(), timeout=.01)
                return diagnostics.failure_context()
            finally:
                diagnostics.current_episode.reset(token)
        contexts = await asyncio.gather(episode(1), episode(2))
        for index, context in enumerate(contexts, 1):
            self.assertEqual(context["sandbox_id"], f"sandbox-{index}")
            self.assertEqual(context["sample_index"], index)
            self.assertEqual(context["failure_phase"], "verification")
            self.assertEqual(context["failure_operation"], "verifier.exec")
            self.assertGreater(context["failure_operation_elapsed_sec"], 0)
        self.assertEqual(diagnostics.failure_context(), {})

    async def test_implicit_error_chain_is_kept_and_secret_redacted(self):
        try:
            try:
                raise BrokenPipeError(32, "test-secret")
            except BrokenPipeError:
                raise httpx.WriteError("")
        except httpx.WriteError as exc:
            with patch.dict(os.environ, {"E2B_API_KEY": "test-secret"}):
                detail = safe_failure_detail(exc, limit=None)
        self.assertIn("BrokenPipeError", detail)
        self.assertIn("WriteError", detail)
        self.assertNotIn("test-secret", detail)

    async def test_grading_failure_records_verifier_phase(self):
        from rl.generate_with_prover import _grade
        state = diagnostics.EpisodeDiagnostics("task", 3, sandbox_id="sandbox-3")
        token = diagnostics.current_episode.set(state)
        async def hang(*_args, **_kwargs):
            await asyncio.Event().wait()
        sandbox = SimpleNamespace(exec=hang, upload_file=AsyncMock())
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(_grade(sandbox, Path(directory)), timeout=.01)
            self.assertEqual(state.phase, "verification")
            self.assertEqual(state.operation, "verifier.exec")
        finally:
            diagnostics.current_episode.reset(token)


    async def test_generation_budget_failure_closes_sandbox_without_grading(self):
        from rl import generate_with_prover as prover
        sample = SimpleNamespace(index=17, response="", metadata={})
        sandbox = SimpleNamespace(_sbx=SimpleNamespace(sandbox_id="sandbox-17"), close=AsyncMock())
        async def hung_episode(*_args):
            diagnostics.set_phase("tool")
            raise prover.GenerationTimeout("generation/tool phase exhausted")
        args = SimpleNamespace(prover_sandbox_concurrency=1, prover_task_root="unused")
        cfg = prover.EpisodeConfig(judge_mode="legacy", model_path="fake", episode_timeout_sec=10)
        fake_types = SimpleNamespace(Sample=Sample)
        fake_output = SimpleNamespace(GenerateFnOutput=SimpleNamespace)
        with patch.dict(sys.modules, {"miles.utils.types": fake_types,
                                     "miles.rollout.base_types": fake_output}), \
             patch.object(prover, "_sandbox_semaphore", None), \
             patch.object(prover, "_episode_config", return_value=cfg), \
             patch.object(prover, "_resolve_task", return_value=(Path("task"), "proof", {})), \
             patch.object(prover, "create_sandbox", AsyncMock(return_value=sandbox)), \
             patch.object(prover, "_router_generate_fn", return_value=None), \
             patch.object(prover, "run_episode", hung_episode):
            output = await prover.generate(SimpleNamespace(args=args, sample=sample, sampling_params={}))
        self.assertIs(output.samples, sample)
        self.assertEqual(sample.metadata["failure_type"], "GenerationTimeout")
        self.assertEqual(sample.metadata["failure_phase"], "tool")
        self.assertEqual(sample.metadata["sandbox_id"], "sandbox-17")
        self.assertEqual(sample.metadata["sample_index"], 17)
        sandbox.close.assert_awaited_once()
        self.assertEqual(diagnostics.failure_context(), {})

    async def test_generate_timeout_metadata_keeps_child_phase_and_sandbox_id(self):
        from rl import generate_with_prover as prover
        sample = SimpleNamespace(index=17, response="", metadata={})
        sandbox = SimpleNamespace(_sbx=SimpleNamespace(sandbox_id="sandbox-17"), close=AsyncMock())
        async def hung_episode(*_args):
            diagnostics.set_phase("verification")
            with diagnostics.operation("verifier.exec"):
                await asyncio.Event().wait()
        args = SimpleNamespace(prover_sandbox_concurrency=1, prover_task_root="unused")
        cfg = prover.EpisodeConfig(judge_mode="legacy", model_path="fake", episode_timeout_sec=.01)
        fake_types = SimpleNamespace(Sample=Sample)
        fake_output = SimpleNamespace(GenerateFnOutput=SimpleNamespace)
        with patch.dict(sys.modules, {"miles.utils.types": fake_types,
                                     "miles.rollout.base_types": fake_output}), \
             patch.object(prover, "_sandbox_semaphore", None), \
             patch.object(prover, "_episode_config", return_value=cfg), \
             patch.object(prover, "_resolve_task", return_value=(Path("task"), "proof", {})), \
             patch.object(prover, "create_sandbox", AsyncMock(return_value=sandbox)), \
             patch.object(prover, "_router_generate_fn", return_value=None), \
             patch.object(prover, "run_episode", hung_episode):
            output = await prover.generate(SimpleNamespace(args=args, sample=sample, sampling_params={}))
        self.assertIs(output.samples, sample)
        self.assertEqual(sample.metadata["failure_type"], "EpisodeTimeout")
        self.assertEqual(sample.metadata["failure_phase"], "verification")
        self.assertEqual(sample.metadata["failure_operation"], "verifier.exec")
        self.assertEqual(sample.metadata["sandbox_id"], "sandbox-17")
        self.assertEqual(sample.metadata["sample_index"], 17)
        sandbox.close.assert_awaited_once()
        self.assertEqual(diagnostics.failure_context(), {})

    async def test_inner_request_timeout_is_not_mislabeled_as_episode_deadline(self):
        from rl import generate_with_prover as prover
        sample = SimpleNamespace(index=17, response="", metadata={})
        sandbox = SimpleNamespace(_sbx=SimpleNamespace(sandbox_id="sandbox-17"), close=AsyncMock())
        async def hung_episode(*_args):
            diagnostics.set_phase("verification")
            with diagnostics.operation("verifier.exec"):
                raise TimeoutError("request deadline")
        args = SimpleNamespace(prover_sandbox_concurrency=1, prover_task_root="unused")
        cfg = prover.EpisodeConfig(judge_mode="legacy", model_path="fake", episode_timeout_sec=3300)
        fake_types = SimpleNamespace(Sample=Sample)
        fake_output = SimpleNamespace(GenerateFnOutput=SimpleNamespace)
        with patch.dict(sys.modules, {"miles.utils.types": fake_types,
                                     "miles.rollout.base_types": fake_output}), \
             patch.object(prover, "_sandbox_semaphore", None), \
             patch.object(prover, "_episode_config", return_value=cfg), \
             patch.object(prover, "_resolve_task", return_value=(Path("task"), "proof", {})), \
             patch.object(prover, "create_sandbox", AsyncMock(return_value=sandbox)), \
             patch.object(prover, "_router_generate_fn", return_value=None), \
             patch.object(prover, "run_episode", hung_episode):
            output = await prover.generate(SimpleNamespace(args=args, sample=sample, sampling_params={}))
        self.assertIs(output.samples, sample)
        self.assertEqual(sample.metadata["failure_type"], "TimeoutError")
        self.assertNotIn("episode exceeded", sample.metadata["failure_detail"])
        self.assertEqual(sample.metadata["failure_phase"], "verification")
        self.assertEqual(sample.metadata["failure_operation"], "verifier.exec")
        self.assertEqual(sample.metadata["sandbox_id"], "sandbox-17")
        self.assertEqual(sample.metadata["sample_index"], 17)
        sandbox.close.assert_awaited_once()
        self.assertEqual(diagnostics.failure_context(), {})


if __name__ == "__main__":
    unittest.main()
