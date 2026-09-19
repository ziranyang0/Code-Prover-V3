from __future__ import annotations

import importlib.util
import json
import os
import pickle
import queue
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rl.generate_with_prover import (
    EpisodeConfig,
    _append_follow_up,
    _cap_tool_result,
    _eligible_for_compile_early_stop,
    _grade,
    _resolve_task,
    _router_generate_fn,
    run_episode,
)
from rl.provenance import prompt_record, validate_prompt_data
from rl.checkpoint_compat import validate_resume_checkpoint
from rl.sandbox import _isolated_async_sandbox_class, e2b_api_options, e2b_creation_options, e2b_connection_env, load_e2b_api_key
from rl.submit_dlc import _job_file_text, _submit_command, build_body


def _task(root: Path, name: str = "task-a") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("prove this\n", encoding="utf-8")
    (task / "environment" / "task.lean").write_text("theorem x : True := by sorry\n")
    (task / "tests" / "test.sh").write_text("true\n")
    (task / "tests" / "grade.py").write_text("print(1)\n")
    return task


class AsyncRLE2BTests(unittest.TestCase):
    def test_qwen_function_xml_is_opt_in_and_decodes_typed_parameters(self):
        from agents import qwen_native_v1 as codec

        content = (
            "checking\n<tool_call>\n<function=Read>\n"
            "<parameter=file_path>\n/task/x.lean\n</parameter>\n"
            "<parameter=limit>\n12\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        strict = codec.decode_assistant(content=content, allowed_tool_names={"Read"})
        self.assertFalse(strict.tool_calls)
        self.assertEqual(strict.errors[0].code, "malformed_tool_json")

        compatible = codec.decode_assistant(
            content=content,
            allowed_tool_names={"Read"},
            call_id_namespace="turn1",
            accept_qwen_function_xml=True,
        )
        self.assertFalse(compatible.errors)
        self.assertEqual(compatible.content, "checking")
        self.assertEqual(len(compatible.tool_calls), 1)
        self.assertEqual(compatible.tool_calls[0].name, "Read")
        self.assertEqual(
            compatible.tool_calls[0].arguments,
            {"file_path": "/task/x.lean", "limit": 12},
        )

    def _write_checkpoint(self, root: Path, tracker: str, **saved_args) -> Path:
        import torch

        (root / "latest_checkpointed_iteration.txt").write_text(tracker, encoding="utf-8")
        iteration_dir = "release" if tracker == "release" else f"iter_{int(tracker):07d}"
        checkpoint = root / iteration_dir
        checkpoint.mkdir()
        torch.save({"args": Namespace(**saved_args)}, checkpoint / "common.pt")
        return root

    def test_numeric_resume_rejects_cross_actor_topology_even_model_only(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_checkpoint(
                Path(directory),
                "1",
                world_size=4,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=4,
                optimizer_offload_fraction=1.0,
            )
            for no_load_optim in (False, True):
                with self.subTest(no_load_optim=no_load_optim):
                    with self.assertRaisesRegex(
                        ValueError,
                        "unsupported cross-actor-topology resume",
                    ):
                        validate_resume_checkpoint(
                            checkpoint,
                            actor_world_size=8,
                            tensor_model_parallel_size=1,
                            pipeline_model_parallel_size=1,
                            context_parallel_size=1,
                            expert_model_parallel_size=4,
                            optimizer_offload_fraction=1.0,
                            no_load_optim=no_load_optim,
                        )

    def test_numeric_resume_accepts_same_actor_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_checkpoint(
                Path(directory),
                "2",
                world_size=8,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=4,
                optimizer_offload_fraction=1.0,
            )
            result = validate_resume_checkpoint(
                checkpoint,
                actor_world_size=8,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=4,
                optimizer_offload_fraction=1.0,
                no_load_optim=False,
            )
            self.assertEqual(result["kind"], "rl_resume")

    def test_numeric_resume_allows_only_cp_change_with_torch_dist(self):
        for checkpoint_format in ("torch_dist", "torch"):
            with self.subTest(checkpoint_format=checkpoint_format), tempfile.TemporaryDirectory() as directory:
                checkpoint = self._write_checkpoint(
                    Path(directory), "29", world_size=32,
                    tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                    context_parallel_size=1, expert_model_parallel_size=4,
                    optimizer_cpu_offload=False, optimizer_offload_fraction=1.0,
                    ckpt_format=checkpoint_format,
                )
                kwargs = dict(actor_world_size=32, tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1, context_parallel_size=4,
                              expert_model_parallel_size=4, optimizer_offload_fraction=0.0,
                              no_load_optim=False)
                if checkpoint_format == "torch_dist":
                    result = validate_resume_checkpoint(checkpoint, **kwargs)
                    self.assertTrue(result["load_optimizer"])
                    self.assertEqual(result["requested_actor_topology"]["context_parallel_size"], 4)
                else:
                    with self.assertRaisesRegex(ValueError, "requires a torch_dist checkpoint"):
                        validate_resume_checkpoint(checkpoint, **kwargs)

    def test_numeric_resume_ignores_fraction_only_when_cpu_offload_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "latest_checkpointed_iteration.txt").write_text("0")
            common = checkpoint / "iter_0000000" / "common.pt"
            common.parent.mkdir()
            common.touch()
            for offload_enabled in (False, True, None):
                saved_args = Namespace(
                    world_size=32,
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    expert_model_parallel_size=4,
                    optimizer_offload_fraction=1.0,
                )
                if offload_enabled is not None:
                    saved_args.optimizer_cpu_offload = offload_enabled
                kwargs = dict(
                    actor_world_size=32,
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    expert_model_parallel_size=4,
                    optimizer_offload_fraction=0.0,
                    no_load_optim=False,
                )
                with self.subTest(offload_enabled=offload_enabled), patch(
                    "rl.checkpoint_compat._load_common_state",
                    return_value={"args": saved_args},
                ):
                    if offload_enabled is False:
                        result = validate_resume_checkpoint(checkpoint, **kwargs)
                        self.assertTrue(result["load_optimizer"])
                        self.assertEqual(result["saved_optimizer_offload_fraction"], 0.0)
                    else:
                        with self.assertRaisesRegex(ValueError, "optimizer offload mismatch"):
                            validate_resume_checkpoint(checkpoint, **kwargs)

    def test_base_release_can_initialize_a_different_actor_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._write_checkpoint(
                Path(directory),
                "release",
                world_size=4,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=4,
                optimizer_offload_fraction=1.0,
            )
            result = validate_resume_checkpoint(
                checkpoint,
                actor_world_size=8,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=4,
                optimizer_offload_fraction=0.9,
                no_load_optim=False,
            )
            self.assertEqual(result["kind"], "base_release")
            self.assertFalse(result["load_optimizer"])

    def test_real_common_state_can_use_explicit_megatron_import_root(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_root = root / "megatron-root"
            package = module_root / "checkpoint_fixture_pkg"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "saved_args.py").write_text(
                "class SavedArgs:\n"
                "    world_size = 4\n"
                "    tensor_model_parallel_size = 1\n"
                "    pipeline_model_parallel_size = 1\n"
                "    context_parallel_size = 1\n"
                "    expert_model_parallel_size = 4\n"
                "    optimizer_offload_fraction = 1.0\n",
                encoding="utf-8",
            )
            # The production guard validates a Megatron-LM checkout before
            # temporarily adding it to sys.path.  Give this fixture the same
            # package shape while keeping the pickled class in a unique module.
            (module_root / "megatron" / "core").mkdir(parents=True)
            (module_root / "megatron" / "__init__.py").write_text("", encoding="utf-8")
            (module_root / "megatron" / "core" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            sys.path.insert(0, str(module_root))
            try:
                from checkpoint_fixture_pkg.saved_args import SavedArgs

                checkpoint = root / "checkpoint"
                (checkpoint / "iter_0000001").mkdir(parents=True)
                (checkpoint / "latest_checkpointed_iteration.txt").write_text(
                    "1", encoding="utf-8"
                )
                torch.save(
                    {"args": SavedArgs()}, checkpoint / "iter_0000001" / "common.pt"
                )
            finally:
                sys.path.remove(str(module_root))
                for name in list(sys.modules):
                    if name == "checkpoint_fixture_pkg" or name.startswith(
                        "checkpoint_fixture_pkg."
                    ):
                        del sys.modules[name]

            result = validate_resume_checkpoint(
                checkpoint,
                actor_world_size=4,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=4,
                optimizer_offload_fraction=1.0,
                no_load_optim=False,
                megatron_lm_root=module_root,
            )
            self.assertEqual(result["kind"], "rl_resume")

    def test_compile_early_stop_source_gate_is_conservative(self):
        self.assertTrue(_eligible_for_compile_early_stop("theorem x : True := by trivial\n"))
        self.assertFalse(_eligible_for_compile_early_stop("theorem x : True := by sorry\n"))
        self.assertFalse(_eligible_for_compile_early_stop("axiom cheat : False\n"))
        self.assertTrue(_eligible_for_compile_early_stop(
            "-- saying sorry in a comment is fine\ntheorem x : True := by trivial\n"
        ))

    def test_grader_infrastructure_failures_are_not_reward_zero(self):
        from rl.sandbox import ExecResult

        class FakeSandbox:
            def __init__(self, results):
                self.results = iter(results)

            async def upload_file(self, *_args, **_kwargs):
                return None

            async def exec(self, *_args, **_kwargs):
                return next(self.results)

        import asyncio

        with tempfile.TemporaryDirectory() as directory:
            tests_dir = Path(directory)
            (tests_dir / "test.sh").write_text("true\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "entrypoint failed"):
                asyncio.run(_grade(FakeSandbox([ExecResult("", "boom", 1)]), tests_dir))
            with self.assertRaisesRegex(RuntimeError, "invalid reward.json"):
                asyncio.run(_grade(FakeSandbox([
                    ExecResult("", "", 0),
                    ExecResult("not-json", "", 0),
                ]), tests_dir))

    def test_tool_follow_up_is_atomic_at_token_budget(self):
        stream = SimpleNamespace(
            tokens=[1, 2, 3],
            loss_mask=[],
            logprobs=[],
            user_turn_tokens=lambda _content: [4, 5, 6],
        )
        self.assertFalse(_append_follow_up(stream, "<tool_response>x</tool_response>", 5))
        self.assertEqual(stream.tokens, [1, 2, 3])
        self.assertTrue(_append_follow_up(stream, "<tool_response>x</tool_response>", 6))
        self.assertEqual(stream.tokens, [1, 2, 3, 4, 5, 6])
        self.assertEqual(stream.loss_mask, [0, 0, 0])

    def test_tool_result_cap_is_token_bounded_and_keeps_head_and_tail(self):
        class CharTokenizer:
            def __call__(self, text, add_special_tokens=False):
                self.assert_no_special = not add_special_tokens
                return {"input_ids": [ord(char) for char in text]}

        result = "HEAD-" + ("x" * 1000) + "-TAIL"
        capped, original_tokens, kept_tokens = _cap_tool_result(
            CharTokenizer(), result, 128
        )
        self.assertEqual(original_tokens, len(result))
        self.assertLessEqual(kept_tokens, 128)
        self.assertTrue(capped.startswith("HEAD-"))
        self.assertTrue(capped.endswith("-TAIL"))
        self.assertIn("tool output truncated", capped)

    def test_prompt_provenance_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            root = tmp_path / "tasks"
            task = _task(root)
            data = tmp_path / "data.jsonl"
            data.write_text(json.dumps(prompt_record(task)) + "\n", encoding="utf-8")
            self.assertEqual(validate_prompt_data(data, root), 1)

            (task / "environment" / "task.lean").write_text("theorem x : True := by trivial\n")
            with self.assertRaisesRegex(ValueError, "task_sha256"):
                validate_prompt_data(data, root)

    def test_generate_resolves_authoritative_instruction_for_token_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tasks"
            task = _task(root)
            record = prompt_record(task)
            sample = SimpleNamespace(prompt=[1, 2, 3], metadata=record["metadata"])
            task_dir, instruction, metadata = _resolve_task(sample, root)
            self.assertEqual(task_dir, task)
            self.assertEqual(instruction, "prove this\n")
            self.assertEqual(metadata["task_sha256"], record["metadata"]["task_sha256"])

    def test_multi_root_prompt_is_contained_and_provenance_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            task = _task(root / "pool-a" / "tasks")
            record = prompt_record(task)
            record["prompt"] = [{"role": "user", "content": record["prompt"]}]
            record["metadata"].update({
                "task_dir": str(task),
                "instruction": "prove this\n",
            })
            data = Path(directory) / "data.jsonl"
            data.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(validate_prompt_data(data, root), 1)

            sample = SimpleNamespace(prompt=[1, 2, 3], metadata=record["metadata"])
            task_dir, instruction, _metadata = _resolve_task(sample, root)
            self.assertEqual(task_dir, task)
            self.assertEqual(instruction, "prove this\n")

            record["metadata"]["task_dir"] = str(Path(directory).parent / task.name)
            data.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the configured task root"):
                validate_prompt_data(data, root)

    def test_router_generate_has_independent_per_turn_token_cap(self):
        import asyncio

        requests = []

        async def fake_post(_url, payload):
            requests.append(payload)
            return {
                "text": "done",
                "meta_info": {
                    "output_token_logprobs": [(-0.1, 7, None)],
                    "finish_reason": {"type": "stop"},
                    "weight_version": "1",
                },
            }

        fake_http = types.ModuleType("miles.utils.http_utils")
        fake_http.post = fake_post
        args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=1)
        with patch.dict(sys.modules, {"miles.utils.http_utils": fake_http}):
            generate_fn = _router_generate_fn(
                args,
                {"max_new_tokens": 32768},
                timeout_sec=30,
                max_tokens_per_turn=4096,
                task_name="task-a",
                sample_index=3,
            )
            output = asyncio.run(generate_fn([1, 2], 20000))
        self.assertEqual(requests[0]["sampling_params"]["max_new_tokens"], 4096)
        self.assertEqual(output["token_ids"], [7])
        self.assertGreaterEqual(output["duration_sec"], 0.0)

    def test_episode_uses_sampled_token_ids_as_authoritative_text_and_tito_prefix(self):
        import asyncio

        from rl import generate_with_prover as prover

        class FakeTokenizer:
            def decode(self, token_ids, **_kwargs):
                self.decoded = list(token_ids)
                return "final answer"

        class FakeTokenStream:
            def __init__(self, _model_path):
                self.tokenizer = FakeTokenizer()
                self.glue = SimpleNamespace(im_end_id=99)
                self.tokens = []
                self.loss_mask = []
                self.logprobs = []
                self.prompt_len = 0

            def start(self, _instruction):
                self.tokens = [1, 2]
                self.prompt_len = 2

            def append_generated(self, token_ids, logprobs):
                self.tokens += token_ids
                self.loss_mask += [1] * len(token_ids)
                self.logprobs += logprobs

            def validate(self):
                self.validated = True

        class FakeSurface:
            async def setup(self, _sandbox):
                return None

            async def _read_task_file(self, _sandbox, _path):
                return "theorem x : True := by sorry\n"

        class FakeSandbox:
            async def upload_file(self, *_args):
                return None

        async def fake_generate(input_ids, _remaining):
            self.assertEqual(input_ids, [1, 2])
            return {
                "token_ids": [7, 99],
                "logprobs": [-0.1, -0.2],
                "finish_reason": "stop",
                # Deliberately contradictory: the episode must ignore this
                # side-channel and parse the exact sampled token IDs instead.
                "text": '<tool_call>{"name":"Bash","arguments":{}}</tool_call>',
                "duration_sec": 0.1,
            }

        async def fake_grade(_sandbox, _tests_dir):
            return 1.0, {"reward": 1.0}

        with tempfile.TemporaryDirectory() as directory:
            task = _task(Path(directory))
            (task / "tests" / "task_file.txt").write_text(
                "/workspace/task.lean\n", encoding="utf-8"
            )
            with patch.object(prover, "TokenStream", FakeTokenStream), patch.object(
                prover, "_ToolSurface", FakeSurface
            ), patch.object(prover, "_grade", fake_grade):
                result = asyncio.run(run_episode(
                    fake_generate,
                    FakeSandbox(),
                    "prove this",
                    task,
                    EpisodeConfig(judge_mode="legacy", model_path="fake", max_turns=1, max_total_tokens=32),
                ))

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_detail, "final_answer")
        self.assertEqual(result.tito_prefix_checks, 1)
        self.assertEqual(result.tokens, [1, 2, 7, 99])

    def test_episode_stops_after_sorry_free_authoritative_file_compiles(self):
        import asyncio

        from rl import generate_with_prover as prover

        class FakeTokenizer:
            def decode(self, _token_ids, **_kwargs):
                return "tool call"

            def __call__(self, text, add_special_tokens=False):
                self.add_special_tokens = add_special_tokens
                return {"input_ids": list(range(len(text)))}

        class FakeTokenStream:
            def __init__(self, _model_path):
                self.tokenizer = FakeTokenizer()
                self.glue = SimpleNamespace(im_end_id=99)
                self.tokens = []
                self.loss_mask = []
                self.logprobs = []
                self.prompt_len = 0

            def start(self, _instruction):
                self.tokens = [1, 2]
                self.prompt_len = 2

            def append_generated(self, token_ids, logprobs):
                self.tokens += token_ids
                self.loss_mask += [1] * len(token_ids)
                self.logprobs += logprobs

            def validate(self):
                return None

        class FakeSurface:
            def __init__(self):
                self.reads = 0

            async def setup(self, _sandbox):
                return None

            async def _read_task_file(self, _sandbox, _path):
                self.reads += 1
                if self.reads == 1:
                    return "theorem x : True := by sorry\n"
                return "theorem x : True := by trivial\n"

            async def _dispatch(self, _sandbox, _name, _arguments):
                return "edited"

            async def _guard_task_file(self, _sandbox, _emit):
                return ""

        class FakeSandbox:
            def __init__(self):
                self.commands = []

            async def upload_file(self, *_args):
                return None

            async def exec(self, command, timeout_sec=None):
                self.commands.append((command, timeout_sec))
                return SimpleNamespace(stdout="", stderr="", return_code=0)

        generate_calls = []

        async def fake_generate(input_ids, _remaining):
            generate_calls.append(list(input_ids))
            return {
                "token_ids": [7, 99],
                "logprobs": [-0.1, -0.2],
                "finish_reason": "stop",
                "duration_sec": 0.1,
            }

        async def fake_grade(_sandbox, _tests_dir):
            return 1.0, {"reward": 1.0}

        decoded = SimpleNamespace(
            errors=[],
            tool_calls=[SimpleNamespace(name="Edit", arguments={"file_path": "/task/x.lean"})],
        )
        sandbox = FakeSandbox()
        with tempfile.TemporaryDirectory() as directory:
            task = _task(Path(directory))
            (task / "tests" / "task_file.txt").write_text(
                "/task/x.lean\n", encoding="utf-8"
            )
            with patch.object(prover, "TokenStream", FakeTokenStream), patch.object(
                prover, "_ToolSurface", FakeSurface
            ), patch.object(prover, "_grade", fake_grade), patch.object(
                prover.codec, "decode_assistant", return_value=decoded
            ):
                result = asyncio.run(run_episode(
                    fake_generate,
                    sandbox,
                    "prove this",
                    task,
                    EpisodeConfig(judge_mode="legacy", model_path="fake", max_turns=10, max_total_tokens=32),
                ))

        self.assertEqual(len(generate_calls), 1)
        self.assertEqual(len(sandbox.commands), 1)
        self.assertIn("cd /task && lake env lean /task/x.lean", sandbox.commands[0][0])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_detail, "compile_passed")
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.early_compile_probes, 1)
        self.assertEqual(result.early_compile_successes, 1)
        self.assertEqual(result.tool_calls, 1)

    def test_episode_limits_consecutive_truncation_nudges(self):
        import asyncio

        from rl import generate_with_prover as prover

        class FakeTokenizer:
            def decode(self, _token_ids, **_kwargs):
                return "unfinished reasoning"

        class FakeTokenStream:
            def __init__(self, _model_path):
                self.tokenizer = FakeTokenizer()
                self.glue = SimpleNamespace(im_end_id=99)
                self.tokens = []
                self.loss_mask = []
                self.logprobs = []
                self.prompt_len = 0

            def start(self, _instruction):
                self.tokens = [1, 2]
                self.prompt_len = 2

            def append_generated(self, token_ids, logprobs):
                self.tokens += token_ids
                self.loss_mask += [1] * len(token_ids)
                self.logprobs += logprobs

            def user_turn_tokens(self, _content):
                return [88]

            def validate(self):
                return None

        class FakeSurface:
            async def setup(self, _sandbox):
                return None

            async def _read_task_file(self, _sandbox, _path):
                return "theorem x : True := by sorry\n"

        class FakeSandbox:
            async def upload_file(self, *_args):
                return None

        calls = []

        async def fake_generate(input_ids, _remaining):
            calls.append(list(input_ids))
            return {
                "token_ids": [7],
                "logprobs": [-0.1],
                "finish_reason": "length",
                "duration_sec": 0.1,
            }

        async def fake_grade(_sandbox, _tests_dir):
            return 0.0, {"reward": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            task = _task(Path(directory))
            (task / "tests" / "task_file.txt").write_text(
                "/workspace/task.lean\n", encoding="utf-8"
            )
            with patch.object(prover, "TokenStream", FakeTokenStream), patch.object(
                prover, "_ToolSurface", FakeSurface
            ), patch.object(prover, "_grade", fake_grade):
                result = asyncio.run(run_episode(
                    fake_generate,
                    FakeSandbox(),
                    "prove this",
                    task,
                    EpisodeConfig(judge_mode="legacy",
                        model_path="fake",
                        max_turns=10,
                        max_total_tokens=32,
                        max_truncation_nudges=1,
                    ),
                ))

        self.assertEqual(len(calls), 2)
        self.assertEqual(result.status, "truncated")
        self.assertEqual(result.stop_detail, "truncated_without_tool_call")
        self.assertEqual(result.tito_prefix_checks, 2)

    def test_episode_walltime_budget_stops_and_still_grades(self):
        import asyncio

        from rl import generate_with_prover as prover

        class FakeTokenStream:
            def __init__(self, _model_path):
                self.tokenizer = object()
                self.tokens = []
                self.loss_mask = []
                self.logprobs = []
                self.prompt_len = 0

            def start(self, _instruction):
                self.tokens = [1, 2]
                self.prompt_len = 2

            def validate(self):
                return None

        class FakeSurface:
            async def setup(self, _sandbox):
                return None

            async def _read_task_file(self, _sandbox, _path):
                return "theorem x : True := by sorry\n"

        class FakeSandbox:
            async def upload_file(self, *_args):
                return None

        async def must_not_generate(*_args):
            raise AssertionError("walltime-expired episode must not call the model")

        grade_calls = []

        async def fake_grade(_sandbox, _tests_dir):
            grade_calls.append(True)
            return 0.25, {"reward": 0.25}

        with tempfile.TemporaryDirectory() as directory:
            task = _task(Path(directory))
            (task / "tests" / "task_file.txt").write_text(
                "/workspace/task.lean\n", encoding="utf-8"
            )
            with patch.object(prover, "TokenStream", FakeTokenStream), patch.object(
                prover, "_ToolSurface", FakeSurface
            ), patch.object(prover, "_grade", fake_grade):
                result = asyncio.run(run_episode(
                    must_not_generate,
                    FakeSandbox(),
                    "prove this",
                    task,
                    EpisodeConfig(judge_mode="legacy", model_path="fake", wall_time_budget_sec=0),
                ))

        self.assertEqual(grade_calls, [True])
        self.assertEqual(result.status, "truncated")
        self.assertEqual(result.stop_detail, "wall_time_budget")
        self.assertEqual(result.reward, 0.25)
        self.assertEqual(result.n_turns, 0)
        self.assertEqual(result.tito_prefix_checks, 0)

    def test_generate_accepts_chat_templated_authoritative_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tasks"
            task = _task(root)
            record = prompt_record(task)
            sample = SimpleNamespace(
                prompt="<|im_start|>user\nprove this\n<|im_end|>\n<|im_start|>assistant\n",
                metadata=record["metadata"],
            )
            task_dir, instruction, metadata = _resolve_task(sample, root)
            self.assertEqual(task_dir, task)
            self.assertEqual(instruction, "prove this\n")
            self.assertEqual(metadata["task_sha256"], record["metadata"]["task_sha256"])

            # The real Qwen HF template applies Jinja ``trim`` to message
            # content, removing instruction.md's final newline.
            sample.prompt = "<|im_start|>user\nprove this<|im_end|>\n<|im_start|>assistant\n"
            task_dir, instruction, metadata = _resolve_task(sample, root)
            self.assertEqual(task_dir, task)
            self.assertEqual(instruction, "prove this\n")
            self.assertEqual(metadata["task_sha256"], record["metadata"]["task_sha256"])

            sample.prompt = "<|im_start|>user\ndifferent task\n<|im_end|>"
            with self.assertRaisesRegex(ValueError, "does not contain"):
                _resolve_task(sample, root)

    def test_generate_success_sets_miles_sample_status(self):
        import asyncio

        from rl import generate_with_prover as prover

        class FakeStatus:
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            FAILED = "failed"

        class FakeMilesSample:
            Status = FakeStatus

        class FakeGenerateFnOutput:
            def __init__(self, samples):
                self.samples = samples

        miles = types.ModuleType("miles")
        rollout = types.ModuleType("miles.rollout")
        base_types = types.ModuleType("miles.rollout.base_types")
        base_types.GenerateFnOutput = FakeGenerateFnOutput
        utils = types.ModuleType("miles.utils")
        miles_types = types.ModuleType("miles.utils.types")
        miles_types.Sample = FakeMilesSample

        class FakeSandbox:
            async def close(self):
                return None

        async def fake_create_sandbox(*_args, **_kwargs):
            return FakeSandbox()

        async def fake_run_episode(*_args, **_kwargs):
            return prover.EpisodeResult(
                tokens=[10, 11, 12],
                prompt_len=1,
                loss_mask=[1, 1],
                logprobs=[-0.1, -0.2],
                status="completed",
                reward=1.0,
                rewards={"proof": 1.0},
                n_turns=1,
                stop_detail="graded",
            )

        class FakeTokenStream:
            def __init__(self, _model_path):
                self.tokenizer = SimpleNamespace(decode=lambda _tokens: "proof")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tasks"
            task = _task(root)
            record = prompt_record(task)
            sample = SimpleNamespace(
                prompt="prove this\n",
                metadata=record["metadata"],
                index=7,
                validate=lambda: None,
            )
            args = SimpleNamespace(
                prover_sandbox_concurrency=1,
                prover_model_path="fake-model",
                prover_judge_mode="legacy",
                prover_max_turns=1,
                prover_max_total_tokens=64,
                prover_max_tokens_per_turn=16,
                prover_max_truncation_nudges=1,
                prover_max_tool_result_tokens=128,
                prover_sandbox_backend="e2b",
                prover_docker_image="unused",
                prover_e2b_template="fake-template",
                prover_wall_time_budget_sec=10,
                prover_episode_timeout_sec=30,
                prover_router_timeout_sec=30,
                prover_task_root=str(root),
                sglang_router_ip="127.0.0.1",
                sglang_router_port=1,
            )
            input_value = SimpleNamespace(args=args, sample=sample, sampling_params={})
            modules = {
                "miles": miles,
                "miles.rollout": rollout,
                "miles.rollout.base_types": base_types,
                "miles.utils": utils,
                "miles.utils.types": miles_types,
            }
            prover._sandbox_semaphore = None
            with patch.dict(sys.modules, modules), patch.object(
                prover, "create_sandbox", fake_create_sandbox
            ), patch.object(prover, "run_episode", fake_run_episode), patch.object(
                prover, "TokenStream", FakeTokenStream
            ), patch.object(
                prover, "_router_generate_fn", return_value=None
            ):
                output = asyncio.run(prover.generate(input_value))

            self.assertIs(output.samples, sample)
            self.assertEqual(sample.status, FakeStatus.COMPLETED)
            self.assertEqual(sample.reward, 1.0)

    def test_e2b_connection_rejects_partial_or_cross_region_fc_configuration(self):
        api = "https://api.ap-southeast-1.e2b.fc.aliyuncs.com"
        domain = "ap-southeast-1.e2b.fc.aliyuncs.com"
        self.assertEqual(e2b_connection_env(api, domain), {
            "E2B_API_URL": api, "E2B_DOMAIN": domain,
        })
        self.assertEqual(e2b_connection_env(), {
            "E2B_API_URL": "https://api.e2b.app", "E2B_DOMAIN": "e2b.app",
        })
        for url, suffix in [
            (api, ""), ("", domain), (api, "e2b.app"),
            ("https://api.e2b.app", domain),
            (api, "cn-beijing.e2b.fc.aliyuncs.com"),
            ("http://api.example.com", "example.com"),
            ("https://user:secret@api.example.com", "example.com"),
            (api + "?api_key=secret", domain),
            (api, "https://" + domain),
        ]:
            with self.subTest(url=url, domain=suffix), self.assertRaises(ValueError):
                e2b_connection_env(url, suffix)

    def test_e2b_gateway_provider_is_carried_to_sdk_api_options(self):
        settings = {
            "E2B_API_URL": "https://api.gateway.example",
            "E2B_DOMAIN": "gateway.example", "E2B_SANDBOX_PROVIDER": "aliyun",
        }
        with patch.dict(os.environ, settings, clear=False):
            self.assertEqual(e2b_api_options(), {
                "api_url": settings["E2B_API_URL"], "domain": settings["E2B_DOMAIN"],
                "api_headers": {"X-Sandbox-Provider": "aliyun"},
            })
        with self.assertRaisesRegex(ValueError, "custom E2B gateway"):
            e2b_connection_env(provider="aliyun")
        with self.assertRaisesRegex(ValueError, "supports aliyun"):
            e2b_connection_env(settings["E2B_API_URL"], settings["E2B_DOMAIN"], "unknown")

    def test_e2b_kruise_metadata_preserves_episode_identity_and_lean_environment(self):
        metadata = {
            "e2b.agents.kruise.io/image": "registry.example/lean@sha256:" + "a" * 64,
            "e2b.agents.kruise.io/create-on-no-stock": "true",
            "run_id": "must-not-override-episode",
        }
        with patch.dict(os.environ, {
            "E2B_CREATE_METADATA_JSON": json.dumps(metadata),
            "E2B_BOOTSTRAP_PROFILE": "lean-base",
        }, clear=False):
            options = e2b_creation_options({"run_id": "actual-run", "sample_index": "42"})
            self.assertEqual(options["metadata"]["run_id"], "actual-run")
            self.assertEqual(options["metadata"]["e2b.agents.kruise.io/create-on-no-stock"], "true")
            self.assertEqual(options["envs"]["ELAN_HOME"], "/root/.elan")
            self.assertIn("/root/.elan/bin", options["envs"]["PATH"])
        with patch.dict(os.environ, {"E2B_CREATE_METADATA_JSON": '{"key": 123}'}, clear=False):
            with self.assertRaisesRegex(ValueError, "string metadata"):
                e2b_creation_options()

    def test_e2b_key_file_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "e2b.key"
            key.write_text("e2b_test_key\n", encoding="utf-8")
            key.chmod(0o600)
            with patch.dict(os.environ, {"E2B_API_KEY_FILE": str(key)}, clear=False):
                os.environ.pop("E2B_API_KEY", None)
                load_e2b_api_key()
                self.assertEqual(os.environ["E2B_API_KEY"], "e2b_test_key")

                key.chmod(0o644)
                with self.assertRaisesRegex(RuntimeError, "0600"):
                    load_e2b_api_key()

    def test_e2b_upload_creates_parent_directory(self):
        from rl.sandbox import E2BSandbox

        calls = []

        class Files:
            async def write(self, target, data):
                calls.append((target, data))

        class RawSandbox:
            files = Files()

        sandbox = E2BSandbox(RawSandbox())

        async def fake_exec(command, timeout=None):
            calls.append(command)
            return SimpleNamespace(stdout="", stderr="", exit_code=0)

        sandbox._sbx.commands = SimpleNamespace(run=fake_exec)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.write_bytes(b"proof")
            import asyncio

            asyncio.run(sandbox.upload_file(source, "/task/example.lean"))
        self.assertEqual(calls[0], "mkdir -p /task")
        self.assertEqual(calls[1], ("/task/example.lean", b"proof"))

    def test_e2b_sandbox_factory_uses_a_distinct_subclass(self):
        try:
            from e2b import AsyncSandbox
        except ModuleNotFoundError:
            self.skipTest("e2b is intentionally optional in the DSW control environment")

        first = _isolated_async_sandbox_class(AsyncSandbox)
        second = _isolated_async_sandbox_class(AsyncSandbox)
        self.assertTrue(issubclass(first, AsyncSandbox))
        self.assertIsNot(first, second)

    def test_e2b_smoke_uses_only_template_baked_files(self):
        from rl import e2b_smoke
        from rl.sandbox import ExecResult

        class FakeSandbox:
            def __init__(self):
                self.commands = []
                self.closed = False

            async def exec(self, command, timeout_sec=None):
                self.commands.append((command, timeout_sec))
                if len(self.commands) == 1:
                    return ExecResult("Lake version test\n", "", 0)
                return ExecResult('{"jsonrpc":"2.0","id":1,"result":{}}\n', "", 0)

            async def upload_file(self, *_args, **_kwargs):
                raise AssertionError("smoke preflight must not upload workspace files")

            async def close(self):
                self.closed = True

        sandbox = FakeSandbox()

        async def fake_create(*_args, **_kwargs):
            return sandbox

        import asyncio

        with patch.object(e2b_smoke, "create_sandbox", fake_create):
            result = asyncio.run(e2b_smoke.smoke("template:id"))
        self.assertTrue(result["ok"])
        self.assertFalse(result["workspace_upload"])
        self.assertTrue(sandbox.closed)
        self.assertEqual(sandbox.commands[0][1], 600)
        self.assertEqual(sandbox.commands[1][1], 120)

    def test_deployment_configuration_requires_explicit_resource_identifiers(self):
        from rl.submit_dlc import _deployment_config
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "--workspace-id"):
                _deployment_config(Namespace(queue="rbe"))
            configured = Namespace(workspace_id="test", resource_id="test",
                                   data_source_id="test", mount_path="relative")
            with self.assertRaisesRegex(ValueError, "absolute"):
                _deployment_config(configured)

    def test_submit_body_is_secret_free_and_has_required_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            codeprover = tmp_path / "codeprover"
            miles = tmp_path / "miles"
            (codeprover / "rl").mkdir(parents=True)
            (codeprover / "rl" / "run_dlc.sh").write_text("#!/bin/bash\n")
            miles.mkdir()
            (miles / "train_async.py").write_text("")
            key = tmp_path / "secret.key"
            key.write_text("do-not-leak")
            key.chmod(stat.S_IRUSR | stat.S_IWUSR)
            args = Namespace(
                name="TRACES_Verification_RL_D3_stage01",
                image="registry.example/codeprover-rl@sha256:" + "f" * 64,
                run_id="test-resume-source",
                nodes=8,
                num_rollout=1,
                save_interval=3,
                optimizer_offload_fraction=0.9,
                no_load_optim=True,
                save_optim=True,
                reset_rollout_data_state=True,
                start_rollout_id=30,
                load_checkpoint_dir="/runs/source/checkpoints",
                rollout_batch_size=32,
                n_samples_per_prompt=2,
                global_batch_size=64,
                over_sampling_multiplier=4,
                zero_group_filtering=True,
                dynamic_sampling_filter_path="rl.filters.check_clean_and_nonzero_std",
                prover_comparator_queue_dir="/shared/data/comparator queue",
                prover_max_total_tokens=32768,
                prover_max_turns=64,
                prover_max_tokens_per_turn=4096,
                prover_max_truncation_nudges=1,
                prover_max_tool_result_tokens=4096,
                prover_router_timeout_sec=300,
                prover_sandbox_concurrency=8,
                prover_wall_time_budget_sec=1200,
                prover_episode_timeout_sec=2400,
                max_seq_len=65536,
                rollout_max_response_len=32768,
                max_tokens_per_gpu=32768,
                log_probs_chunk_size=256,
                sglang_server_concurrency=2,
                prover_async_queue_size=128,
                prover_async_pause_timeout_sec=300,
                router_balance_abs_threshold=1,
                queue="rbe",
                workspace_id="test-workspace",
                resource_id="test-resource",
                data_source_id="test-dataset",
                mount_path="/shared/data",
                codeprover_root=str(codeprover),
                miles_root=str(miles),
                e2b_key_file=str(key),
                prompt_data="/data/train-5000.jsonl",
                task_root="/data/tasks-5000",
                load_debug_rollout_data="/data/rollout-0.pt",
                codeprover_commit="a" * 40,
                codeprover_patch_sha256="c" * 64,
                miles_commit="b" * 40,
                miles_patch_sha256="d" * 64,
                hf_checkpoint="/model/hf",
                megatron_checkpoint="/model/megatron",
                megatron_lm_root="/src/megatron",
                e2b_template="template:id",
                run_root="/runs",
                max_running_minutes=60,
                attempt_id="attempt-123",
                topic_id="test-topic",
            )
            body = build_body(args)
            self.assertEqual(body["Envs"]["PROVER_JUDGE_MODE"], "comparator")
            self.assertEqual(body["Envs"]["PROVER_COMPARATOR_QUEUE_DIR"], "/shared/data/comparator queue")
            self.assertTrue(body["Envs"]["PROVER_PROOF_ARTIFACTS_DIR"].endswith("/test-resume-source/proofs"))
            encoded = json.dumps(body)
            self.assertNotIn("do-not-leak", encoded)
            self.assertEqual(body["WorkspaceId"], "test-workspace")
            self.assertEqual(body["ResourceId"], "test-resource")
            self.assertEqual(body["Priority"], 9)
            self.assertEqual(body["JobSpecs"][0]["PodCount"], 8)
            self.assertEqual(body["JobSpecs"][0]["ResourceConfig"]["GPU"], "8")
            self.assertEqual(body["JobSpecs"][0]["Image"], args.image)
            self.assertEqual(body["JobSpecs"][0]["ResourceConfig"]["CPU"], "192")
            self.assertEqual(body["JobSpecs"][0]["ResourceConfig"]["Memory"], "1800Gi")
            self.assertNotIn("UserVpc", body)
            self.assertEqual(body["DataSources"][0], {
                "DataSourceId": "test-dataset",
                "DataSourceVersion": "v1",
                "MountPath": "/shared/data",
            })
            self.assertIs(body["Settings"]["EnableRDMA"], True)
            self.assertEqual(body["Settings"]["Tags"]["sandbox"], "e2b")
            self.assertEqual(body["Envs"]["E2B_API_KEY_FILE"], str(key))
            self.assertEqual(body["Envs"]["ATTEMPT_ID"], "attempt-123")
            self.assertEqual(
                body["Envs"]["RUN_ID"],
                "test-resume-source",
            )
            self.assertEqual(body["Envs"]["SAVE_INTERVAL"], "3")
            self.assertEqual(body["Envs"]["OPTIMIZER_OFFLOAD_FRACTION"], "0.9")
            self.assertEqual(body["Envs"]["NO_LOAD_OPTIM"], "1")
            self.assertEqual(body["Envs"]["SAVE_OPTIM"], "1")
            self.assertEqual(body["Envs"]["RESET_ROLLOUT_DATA_STATE"], "1")
            self.assertEqual(body["Envs"]["START_ROLLOUT_ID"], "30")
            self.assertEqual(
                body["Envs"]["LOAD_CHECKPOINT_DIR"],
                "/runs/source/checkpoints",
            )
            self.assertEqual(body["Envs"]["GPUS_PER_NODE"], "8")
            self.assertEqual(body["Envs"]["CODEPROVER_PATCH_SHA256"], "c" * 64)
            self.assertEqual(body["Envs"]["MILES_PATCH_SHA256"], "d" * 64)
            self.assertEqual(body["Envs"]["PROMPT_DATA"], "/data/train-5000.jsonl")
            self.assertEqual(body["Envs"]["TASK_ROOT"], "/data/tasks-5000")
            self.assertEqual(body["Envs"]["LOAD_DEBUG_ROLLOUT_DATA"], "/data/rollout-0.pt")
            self.assertEqual(body["Envs"]["ROLLOUT_BATCH_SIZE"], "32")
            self.assertEqual(body["Envs"]["GLOBAL_BATCH_SIZE"], "64")
            self.assertEqual(body["Envs"]["OVER_SAMPLING_BATCH_SIZE"], "128")
            self.assertEqual(body["Envs"]["PROVER_MAX_TOTAL_TOKENS"], "32768")
            self.assertEqual(body["Envs"]["PROVER_MAX_TURNS"], "64")
            self.assertEqual(body["Envs"]["PROVER_MAX_TOKENS_PER_TURN"], "4096")
            self.assertEqual(body["Envs"]["PROVER_MAX_TRUNCATION_NUDGES"], "1")
            self.assertEqual(body["Envs"]["PROVER_MAX_TOOL_RESULT_TOKENS"], "4096")
            self.assertEqual(body["Envs"]["PROVER_WALL_TIME_BUDGET_SEC"], "1200")
            self.assertEqual(body["Envs"]["PROVER_EPISODE_TIMEOUT_SEC"], "2400")
            self.assertEqual(body["Envs"]["PROVER_ROUTER_TIMEOUT_SEC"], "300")
            self.assertEqual(body["Envs"]["PROVER_SANDBOX_CONCURRENCY"], "8")
            self.assertEqual(body["Envs"]["MAX_SEQ_LEN"], "65536")
            self.assertEqual(body["Envs"]["ROLLOUT_MAX_RESPONSE_LEN"], "32768")
            self.assertEqual(body["Envs"]["MAX_TOKENS_PER_GPU"], "32768")
            self.assertEqual(body["Envs"]["LOG_PROBS_CHUNK_SIZE"], "256")
            self.assertEqual(
                body["Envs"]["PYTORCH_CUDA_ALLOC_CONF"],
                "expandable_segments:True",
            )
            self.assertEqual(body["Envs"]["SGLANG_SERVER_CONCURRENCY"], "2")
            self.assertEqual(body["Envs"]["PROVER_ASYNC_QUEUE_SIZE"], "128")
            self.assertEqual(body["Envs"]["PROVER_ASYNC_PAUSE_TIMEOUT_SEC"], "300")
            self.assertEqual(body["Envs"]["ROUTER_BALANCE_ABS_THRESHOLD"], "1")
            self.assertEqual(
                body["Envs"]["DYNAMIC_SAMPLING_FILTER_PATH"],
                "rl.filters.check_clean_and_nonzero_std",
            )
            self.assertEqual(
                body["Settings"]["Tags"]["topic_id"],
                "test-topic",
            )
            params = _job_file_text(body)
            self.assertIn("workspace_id=test-workspace\n", params)
            self.assertIn("workers=8\n", params)
            self.assertNotIn("do-not-leak", params)
            submit = _submit_command(body)
            self.assertEqual(submit[:3], [
                "/etc/dsw/runtime/export_bin/aliyun", "pai-dlc", "CreateJob"
            ])
            self.assertIn("ap-southeast-1", submit)
            self.assertIn("pai-dlc.ap-southeast-1.aliyuncs.com", submit)
            sent = json.loads(submit[submit.index("--body") + 1])
            self.assertEqual(sent, body)
            self.assertTrue(sent["Settings"]["EnableRDMA"])
            self.assertEqual(sent["Envs"]["NCCL_MNNVL_ENABLE"], "0")
            self.assertEqual(sent["Envs"]["NCCL_ALGO"], "^NVLS")
            self.assertEqual(sent["Envs"]["TORCHDYNAMO_DISABLE"], "1")
            self.assertEqual(sent["JobSpecs"][0]["PodCount"] * int(
                sent["JobSpecs"][0]["ResourceConfig"]["GPU"]
            ), 64)
            self.assertNotIn("HyperNode", json.dumps(sent))

            args.e2b_api_url = "https://api.ap-southeast-1.e2b.fc.aliyuncs.com"
            args.e2b_domain = "ap-southeast-1.e2b.fc.aliyuncs.com"
            args.e2b_template = "fc-lean428-fixture"
            fc_body = build_body(args)
            self.assertEqual(fc_body["Envs"]["E2B_API_URL"], args.e2b_api_url)
            self.assertEqual(fc_body["Envs"]["E2B_DOMAIN"], args.e2b_domain)
            self.assertEqual(fc_body["Envs"]["E2B_TEMPLATE_ID"], args.e2b_template)
            self.assertNotIn("do-not-leak", json.dumps(fc_body))
            args.e2b_domain = "cn-beijing.e2b.fc.aliyuncs.com"
            with self.assertRaisesRegex(ValueError, "same region"):
                build_body(args)
            args.e2b_domain = "ap-southeast-1.e2b.fc.aliyuncs.com"
            args.e2b_template = ""
            with self.assertRaisesRegex(ValueError, "template built on that service"):
                build_body(args)
            args.e2b_api_url = "https://api.gateway.example"
            args.e2b_domain = "gateway.example"
            args.e2b_sandbox_provider = "aliyun"
            args.e2b_template = "ali-lean428-fixture"
            gateway_body = build_body(args)
            self.assertEqual(gateway_body["Envs"]["E2B_SANDBOX_PROVIDER"], "aliyun")
            self.assertEqual(gateway_body["Envs"]["E2B_API_URL"], args.e2b_api_url)
            self.assertNotIn("do-not-leak", json.dumps(gateway_body))
            args.e2b_sandbox_provider = ""
            args.e2b_api_url = "https://api.sandbox.example"
            args.e2b_domain = "sandbox.example"
            args.e2b_validate_api_key = False
            args.e2b_create_metadata_json = '{"e2b.agents.kruise.io/create-on-no-stock":"true"}'
            args.e2b_create_timeout_sec = 960
            args.e2b_bootstrap_profile = "lean-base"
            import ssl
            args.e2b_ca_bundle = ssl.get_default_verify_paths().cafile
            private_sg_body = build_body(args)
            self.assertEqual(private_sg_body["Envs"]["E2B_VALIDATE_API_KEY"], "false")
            self.assertEqual(private_sg_body["Envs"]["E2B_CREATE_TIMEOUT_SEC"], "960")
            self.assertEqual(private_sg_body["Envs"]["E2B_BOOTSTRAP_PROFILE"], "lean-base")
            self.assertEqual(json.loads(private_sg_body["Envs"]["E2B_CREATE_METADATA_JSON"]),
                             {"e2b.agents.kruise.io/create-on-no-stock": "true"})
            if args.e2b_ca_bundle:
                self.assertEqual(private_sg_body["Envs"]["SSL_CERT_FILE"],
                                 str(Path(args.e2b_ca_bundle).resolve()))
            self.assertNotIn("E2B_SANDBOX_PROVIDER", private_sg_body["Envs"])
            args.e2b_ca_bundle = str(tmp_path / "missing.pem")
            with self.assertRaisesRegex(ValueError, "CA bundle"):
                build_body(args)
            args.e2b_ca_bundle = None
            args.e2b_validate_api_key = True
            args.e2b_api_url = args.e2b_domain = ""
            args.e2b_template = "template:id"

            args.nodes = 10
            expanded = build_body(args)
            self.assertEqual(expanded["JobSpecs"][0]["PodCount"], 10)
            self.assertEqual(expanded["Envs"]["NNODES"], "10")
            self.assertEqual(expanded["Settings"]["Tags"]["nodes"], "10")
            self.assertEqual(expanded["JobSpecs"][0]["ResourceConfig"], body["JobSpecs"][0]["ResourceConfig"])
            args.nodes = 8

            args.name = "test-03"
            with self.assertRaisesRegex(ValueError, "TRACES_Verification_RL_"):
                build_body(args)
            args.name = "TRACES_Verification_RL_D3_stage01"
            args.nodes = 16
            with self.assertRaisesRegex(ValueError, "eight-GPU H200"):
                build_body(args)
            args.nodes = 8
            args.image = "registry.example/codeprover-rl:latest"
            self.assertEqual(build_body(args)["JobSpecs"][0]["Image"], args.image)
            args.submit = True
            with self.assertRaisesRegex(ValueError, "digest"):
                build_body(args)

    def test_h200_launcher_passes_8x8_layout_to_preflight_and_ray(self):
        launcher = Path(__file__).parents[1] / "rl" / "run_prover_rl.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            (repo / "rl").mkdir(parents=True)
            (repo / "rl" / "preflight.py").write_text(
                "import json, sys; print(json.dumps(sys.argv[1:]))\n"
            )
            (repo / "rl" / "e2b_smoke.py").write_text("print('{}')\n")
            miles = root / "miles"
            models = miles / "scripts" / "models"
            models.mkdir(parents=True)
            (models / "qwen3.6-35B-A3B.sh").write_text("MODEL_ARGS=()\n")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            ray = bin_dir / "ray"
            ray.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "print(json.dumps(sys.argv[1:]))\n"
            )
            ray.chmod(0o755)
            env = dict(os.environ)
            env.update({
                "PATH": str(bin_dir) + os.pathsep + env["PATH"],
                "MILES_ROOT": str(miles),
                "CODEPROVER_ROOT": str(repo),
                "MEGATRON_LM_ROOT": str(root / "megatron"),
                "HF_CHECKPOINT": str(root / "hf"),
                "MEGATRON_CHECKPOINT": str(root / "release"),
                "RUN_DIR": str(root / "run"),
                "E2B_API_KEY_FILE": str(root / "key"),
                "E2B_TEMPLATE_ID": "test-lean-template",
                "CODEPROVER_SOURCE_COMMIT": "a" * 40,
                "MILES_SOURCE_COMMIT": "b" * 40,
                "CODEPROVER_PATCH_SHA256": "c" * 64,
                "MILES_PATCH_SHA256": "d" * 64,
                "NNODES": "8", "GPUS_PER_NODE": "8",
                "ROLLOUT_BATCH_SIZE": "32", "N_SAMPLES_PER_PROMPT": "8",
                "GLOBAL_BATCH_SIZE": "256", "NO_LOAD_OPTIM": "1",
                "OPTIMIZER_OFFLOAD_FRACTION": "0.0",
                "RUN_ID": "TRACES_Verification_RL_fixture",
                "PROVER_COMPARATOR_QUEUE_DIR": str(root / "comparator queue"),
            })
            result = subprocess.run(
                ["bash", str(launcher)], env=env, check=True,
                capture_output=True, text=True,
            )
            argv = json.loads(result.stdout)
            for flag, expected in {
                "--actor-num-nodes": "4",
                "--actor-num-gpus-per-node": "8",
                "--num-gpus-per-node": "8",
                "--rollout-num-gpus": "32",
                "--rollout-num-gpus-per-engine": "1",
                "--expert-model-parallel-size": "4",
                "--global-batch-size": "256",
            }.items():
                self.assertEqual(argv[argv.index(flag) + 1], expected)
            self.assertEqual(argv[argv.index("--prover-judge-mode") + 1], "comparator")
            self.assertEqual(argv[argv.index("--prover-comparator-queue-dir") + 1], str(root / "comparator queue"))
            self.assertEqual(argv[argv.index("--prover-proof-artifacts-dir") + 1], str(root / "run" / "proofs"))
            self.assertNotIn("--sglang-attention-backend", argv)
            runtime = json.loads(
                next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json="))
            )
            self.assertEqual(runtime["env_vars"]["NCCL_MNNVL_ENABLE"], "0")
            for key, expected in {
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_CUMEM_ENABLE": "1", "NCCL_ALGO": "^NVLS",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TORCHDYNAMO_DISABLE": "1", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            }.items():
                self.assertEqual(runtime["env_vars"][key], expected)
            preflight = json.loads((root / "run" / "preflight.json").read_text())
            for flag in ("--prover-judge-mode", "--prover-comparator-queue-dir", "--prover-proof-artifacts-dir"):
                self.assertEqual(preflight[preflight.index(flag) + 1], argv[argv.index(flag) + 1])
            self.assertEqual(preflight[preflight.index("--nodes") + 1], "8")
            self.assertEqual(preflight[preflight.index("--actor-gpus") + 1], "32")
            self.assertEqual(preflight[preflight.index("--rollout-gpus") + 1], "32")

            self.assertNotIn("--eval-interval", argv)
            self.assertNotIn("--eval-prompt-data", argv)
            train_path, eval_path = root / "train.jsonl", root / "fixed eval.jsonl"
            for path, name in ((train_path, "train"), (eval_path, "eval")):
                path.write_text(json.dumps({"metadata": {
                    "task_name": name, "task_sha256": name,
                }}) + "\n")
            env.update({
                "PYTHONPATH": str(launcher.parents[1]),
                "PROMPT_DATA": str(train_path),
                "PROVER_EVAL_PROMPT_DATA": str(eval_path),
                "PROVER_TUNING_PROMPT_DATA": "",
                "EVAL_INTERVAL": "10",
            })
            env.pop("N_SAMPLES_PER_EVAL_PROMPT", None)
            env.pop("PROVER_FULL_EVAL_SAMPLES_PER_PROMPT", None)
            result = subprocess.run(
                ["bash", str(launcher)], env=env, check=True,
                capture_output=True, text=True,
            )
            argv = json.loads(result.stdout)
            dataset_index = argv.index("--eval-prompt-data")
            self.assertEqual(argv[dataset_index + 1:dataset_index + 3], ["d3_hard", str(eval_path)])
            for flag, expected in {
                "--eval-function-path": "rl.evaluate.generate_rollout_eval",
                "--prover-eval-prompt-data": str(eval_path),
                "--eval-interval": "10", "--n-samples-per-eval-prompt": "2",
                "--prover-full-eval-samples-per-prompt": "8",
            }.items():
                self.assertEqual(argv[argv.index(flag) + 1], expected)

            env.update({
                "E2B_API_URL": "https://api.gateway.example",
                "E2B_DOMAIN": "gateway.example",
                "E2B_SANDBOX_PROVIDER": "aliyun",
                "E2B_TEMPLATE_ID": "fc-lean428-fixture",
                "E2B_CREATE_CONCURRENCY": "8",
                "E2B_WARM_POOL_SIZE": "32", "E2B_WARM_POOL_WAIT_TIMEOUT_SEC": "1500",
                "E2B_API_KEY": "fixture-must-not-be-in-ray-runtime-json",
                "NNODES": "10", "CONTEXT_PARALLEL_SIZE": "8",
                "PROVER_MAX_TURNS": "96", "PROVER_MAX_TOKENS_PER_TURN": "16384",
                "PROVER_MAX_TOTAL_TOKENS": "131072", "MAX_SEQ_LEN": "131072",
                "ROLLOUT_MAX_RESPONSE_LEN": "131072", "MAX_TOKENS_PER_GPU": "16384",
                "PROVER_SANDBOX_CONCURRENCY": "384", "SGLANG_SERVER_CONCURRENCY": "8",
            })
            result = subprocess.run(
                ["bash", str(launcher)], env=env, check=True,
                capture_output=True, text=True,
            )
            argv = json.loads(result.stdout)
            for flag, expected in {
                "--actor-num-nodes": "4", "--actor-num-gpus-per-node": "8",
                "--rollout-num-gpus": "48", "--rollout-num-gpus-per-engine": "1",
                "--context-parallel-size": "8", "--expert-model-parallel-size": "4",
                "--prover-max-turns": "96", "--prover-sandbox-concurrency": "384",
                "--prover-max-total-tokens": "131072", "--global-batch-size": "256",
            }.items():
                self.assertEqual(argv[argv.index(flag) + 1], expected)
            self.assertEqual(argv[argv.index("--prover-judge-mode") + 1], "comparator")
            self.assertEqual(argv[argv.index("--prover-comparator-queue-dir") + 1], str(root / "comparator queue"))
            self.assertEqual(argv[argv.index("--prover-proof-artifacts-dir") + 1], str(root / "run" / "proofs"))
            self.assertNotIn("--sglang-attention-backend", argv)
            self.assertIn("--eval-prompt-data", argv)
            preflight = json.loads((root / "run" / "preflight.json").read_text())
            runtime = json.loads(next(
                arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")
            ))
            for name in ("E2B_API_URL", "E2B_DOMAIN", "E2B_SANDBOX_PROVIDER", "E2B_API_KEY_FILE", "E2B_CREATE_CONCURRENCY", "E2B_WARM_POOL_SIZE", "E2B_WARM_POOL_WAIT_TIMEOUT_SEC", "RUN_DIR"):
                self.assertEqual(runtime["env_vars"][name], env[name])
            self.assertNotIn("E2B_API_KEY", runtime["env_vars"])
            self.assertNotIn("fixture-must-not-be-in-ray-runtime-json", json.dumps(argv))
            self.assertEqual(argv[argv.index("--prover-e2b-template") + 1], "fc-lean428-fixture")
            manifest = (root / "run" / "run-manifest.txt").read_text()
            self.assertIn("e2b_api_url=" + env["E2B_API_URL"], manifest)
            self.assertIn("e2b_domain=" + env["E2B_DOMAIN"], manifest)
            for flag, expected in {"--nodes": "10", "--actor-gpus": "32",
                                   "--rollout-gpus": "48", "--actor-context-parallel-size": "8"}.items():
                self.assertEqual(preflight[preflight.index(flag) + 1], expected)

            env.update({
                "E2B_API_URL": "https://api.sandbox.example", "E2B_DOMAIN": "sandbox.example",
                "E2B_SANDBOX_PROVIDER": "", "E2B_VALIDATE_API_KEY": "false",
                "SSL_CERT_FILE": "/mounted/sg-ca-bundle.pem", "NO_PROXY": "*",
                "E2B_CREATE_METADATA_JSON": '{"e2b.agents.kruise.io/create-on-no-stock":"true"}',
                "E2B_CREATE_TIMEOUT_SEC": "960", "E2B_BOOTSTRAP_PROFILE": "lean-base",
                "E2B_TEMPLATE_ID": "private-sg-lean428-fixture",
            })
            result = subprocess.run(
                ["bash", str(launcher)], env=env, check=True,
                capture_output=True, text=True,
            )
            argv = json.loads(result.stdout)
            runtime = json.loads(next(
                arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")
            ))
            for name in ("E2B_API_URL", "E2B_DOMAIN", "E2B_VALIDATE_API_KEY", "SSL_CERT_FILE", "NO_PROXY", "E2B_CREATE_METADATA_JSON", "E2B_CREATE_TIMEOUT_SEC", "E2B_BOOTSTRAP_PROFILE"):
                self.assertEqual(runtime["env_vars"][name], env[name])
            self.assertNotIn("E2B_API_KEY", runtime["env_vars"])
            self.assertEqual(argv[argv.index("--prover-e2b-template") + 1], "private-sg-lean428-fixture")

    def test_main_rl_environment_keeps_harbor_isolated(self):
        root = Path(__file__).parents[1]
        requirements = (root / "requirements-rl.txt").read_text(encoding="utf-8")
        entrypoint = (root / "rl" / "run_dlc.sh").read_text(encoding="utf-8")
        self.assertIn("e2b==2.34.0", requirements)
        self.assertNotIn("harbor==", requirements)
        self.assertNotIn("pip install", entrypoint)
        self.assertIn('{"e2b": "2.34.0", "openai": "2.6.1"}', entrypoint)
        self.assertIn('importlib.metadata.version("harbor")', entrypoint)

    def test_qwen_tool_surface_imports_without_harbor(self):
        from agents.qwen_native_agent import QwenNativeAgent

        with tempfile.TemporaryDirectory() as directory:
            agent = QwenNativeAgent(Path(directory), model_name="fixture")
        self.assertEqual(agent.model_name, "fixture")

    def test_qwen_tool_surface_corrects_non_authoritative_work_path(self):
        import asyncio

        from agents.qwen_native_agent import QwenNativeAgent

        class FakeEnvironment:
            async def exec(self, *_args, **_kwargs):
                raise AssertionError("a /work command must not be executed")

        with tempfile.TemporaryDirectory() as directory:
            agent = QwenNativeAgent(Path(directory), model_name="fixture")
            agent._guard_path = "/task/example.lean"
            result = asyncio.run(agent._dispatch(
                FakeEnvironment(),
                "Bash",
                {"command": "ls -la /work/example.lean"},
            ))
            task_result = asyncio.run(agent._dispatch(FakeEnvironment(), "Task", {}))

        self.assertIn("was NOT executed", result)
        self.assertIn("/task/example.lean", result)
        self.assertIn("subagents are unavailable", task_result)

    def test_async_queue_pop_preserves_surplus(self):
        fake_rollout = types.ModuleType("miles.rollout.sglang_rollout")
        fake_rollout.GenerateState = object
        fake_rollout.generate_and_rm_group = lambda *args, **kwargs: None
        fake_async = types.ModuleType("miles.utils.async_utils")
        fake_async.run = lambda value: value

        class FakeSample:
            class Status:
                COMPLETED = "completed"
                TRUNCATED = "truncated"
                FAILED = "failed"
                ABORTED = "aborted"

        fake_types = types.ModuleType("miles.utils.types")
        fake_types.Sample = FakeSample
        module_name = "_test_fully_async_rollout"
        path = Path(__file__).parents[1] / "rl" / "fully_async_rollout.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "miles": types.ModuleType("miles"),
            "miles.rollout": types.ModuleType("miles.rollout"),
            "miles.rollout.base_types": SimpleNamespace(RolloutFnTrainOutput=lambda **kw: SimpleNamespace(**kw)),
            "miles.rollout.sglang_rollout": fake_rollout,
            "miles.utils": types.ModuleType("miles.utils"),
            "miles.utils.async_utils": fake_async,
            "miles.utils.types": fake_types,
        }):
            assert spec.loader is not None
            spec.loader.exec_module(module)
        worker = module.AsyncRolloutWorker.__new__(module.AsyncRolloutWorker)
        worker.output_queue = queue.Queue()
        worker.output_queue.put((1, ["first"]))
        worker.output_queue.put((2, ["second"]))
        self.assertEqual(worker.pop_completed(), (1, ["first"]))
        self.assertEqual(worker.pop_completed(), (2, ["second"]))

        recycled = []

        class Resettable:
            def __init__(self):
                self.reset = False
                self.status = FakeSample.Status.ABORTED

            def reset_for_retry(self):
                self.reset = True

        sample = Resettable()
        worker.data_source = SimpleNamespace(retry_inflight=lambda group: recycled.append(group))

        async def cancel_read_ahead():
            import asyncio

            task = asyncio.create_task(asyncio.sleep(60))
            active = {task: (3, [sample])}
            await worker._cancel_for_pause(active)
            self.assertEqual(active, {})

        import asyncio

        asyncio.run(cancel_read_ahead())
        self.assertTrue(sample.reset)
        self.assertEqual(recycled, [[sample]])

    def test_async_zero_group_filter_discards_before_accepting(self):
        fake_rollout = types.ModuleType("miles.rollout.sglang_rollout")
        fake_rollout.GenerateState = object
        fake_rollout.generate_and_rm_group = lambda *args, **kwargs: None
        fake_async = types.ModuleType("miles.utils.async_utils")
        fake_async.run = lambda value: value

        class FakeSample:
            class Status:
                COMPLETED = "completed"
                TRUNCATED = "truncated"
                FAILED = "failed"
                ABORTED = "aborted"

        fake_types = types.ModuleType("miles.utils.types")
        fake_types.Sample = FakeSample
        fake_misc = types.ModuleType("miles.utils.misc")
        fake_misc.load_function = lambda _path: (
            lambda _args, group: SimpleNamespace(
                keep=len({sample.reward for sample in group}) > 1,
                reason="zero_std" if len({sample.reward for sample in group}) == 1 else None,
            )
        )
        fake_filter = types.ModuleType("miles.rollout.filter_hub.base_types")
        fake_filter.call_dynamic_filter = lambda fn, *args, **kwargs: fn(*args, **kwargs)

        path = Path(__file__).parents[1] / "rl" / "fully_async_rollout.py"
        spec = importlib.util.spec_from_file_location("_test_fully_async_filter", path)
        module = importlib.util.module_from_spec(spec)
        modules = {
            "miles": types.ModuleType("miles"),
            "miles.rollout": types.ModuleType("miles.rollout"),
            "miles.rollout.base_types": SimpleNamespace(RolloutFnTrainOutput=lambda **kw: SimpleNamespace(**kw)),
            "miles.rollout.sglang_rollout": fake_rollout,
            "miles.rollout.filter_hub": types.ModuleType("miles.rollout.filter_hub"),
            "miles.rollout.filter_hub.base_types": fake_filter,
            "miles.utils": types.ModuleType("miles.utils"),
            "miles.utils.async_utils": fake_async,
            "miles.utils.misc": fake_misc,
            "miles.utils.types": fake_types,
        }
        with patch.dict(sys.modules, modules):
            assert spec.loader is not None
            spec.loader.exec_module(module)

            def sample(index, reward):
                return SimpleNamespace(
                    index=index,
                    reward=reward,
                    status="completed",
                    oldest_weight_version=None,
                    prompt="prompt",
                    response="response",
                    label="label",
                )

            def failed(index):
                result = sample(index, 0.0)
                result.status = "failed"
                result.metadata = {"failure_type": "temporary"}
                result.reset_for_retry = lambda: None
                return result

            class Worker:
                def __init__(self):
                    self.completed = iter([
                        (10, [failed(10), failed(10)]),
                        (1, [sample(1, 0.0), sample(1, 0.0)]),
                        (11, [failed(11), failed(11)]),
                        (2, [sample(2, 0.0), sample(2, 1.0)]),
                    ])
                    self.pause_timeout = None
                    self.stopped = False

                def raise_if_failed(self):
                    return None

                def pop_completed(self):
                    return next(self.completed, None)

                def pause(self, timeout_sec):
                    self.pause_timeout = timeout_sec

                def stop(self):
                    self.stopped = True

            class DataSource:
                def __init__(self):
                    self.discarded = []
                    self.accepted = []
                    self.retried = []

                def retry_completed(self, group_id, group):
                    self.retried.append(group_id)

                def discard_completed(self, group_id):
                    self.discarded.append(group_id)

                def record_accepted(self, rollout_id, group_id):
                    self.accepted.append((rollout_id, group_id))

            args = SimpleNamespace(
                rollout_global_dataset=True,
                rollout_batch_size=1,
                n_samples_per_prompt=2,
                dynamic_sampling_filter_path="test.filter",
                max_weight_staleness=None,
                prover_async_no_progress_timeout_sec=1,
                prover_async_max_consecutive_failures=2,
                prover_async_pause_timeout_sec=2,
                num_rollout=1,
            )
            data_source = DataSource()
            import asyncio

            worker = Worker()
            with patch.object(module, "get_global_worker", return_value=worker):
                result = asyncio.run(module.generate_rollout_async(args, 7, data_source))

        self.assertEqual([[sample.reward for sample in group] for group in result.samples], [[0.0, 1.0]])
        self.assertEqual(result.metrics["sampling/raw/groups"], 2)
        self.assertEqual(result.metrics["sampling/raw/pass@1"], .25)
        self.assertEqual(result.metrics["sampling/accepted/pass@1"], .5)
        self.assertEqual(result.metrics["sampling/accepted_group_fraction"], .5)
        self.assertEqual(result.metrics["sampling/infrastructure_failed_groups"], 2)
        self.assertEqual(data_source.discarded, [1])
        self.assertEqual(data_source.retried, [10, 11])
        self.assertEqual(data_source.accepted, [(7, 2)])
        self.assertEqual(worker.pause_timeout, 2.0)
        self.assertTrue(worker.stopped)

    def test_checkpoint_buffer_keeps_only_untrained_read_ahead(self):
        class FakeTorch(types.ModuleType):
            @staticmethod
            def save(value, path):
                with open(path, "wb") as handle:
                    pickle.dump(value, handle)

            @staticmethod
            def load(path, weights_only=False):
                del weights_only
                with open(path, "rb") as handle:
                    return pickle.load(handle)

        class FakeBase:
            def __init__(self, args):
                self.args = args
                self.max_prompt_len_during_init = getattr(args, "rollout_max_prompt_len", None)
                self.buffer = []

            def add_samples(self, samples):
                self.buffer.extend(samples)

            def save(self, rollout_id):
                del rollout_id

            def load(self, rollout_id=None):
                del rollout_id

        fake_data_source = types.ModuleType("miles.rollout.data_source")
        fake_data_source.RolloutDataSourceWithBuffer = FakeBase
        fake_async_rollout = types.ModuleType("rl.fully_async_rollout")
        fake_async_rollout.get_existing_worker = lambda _source: None
        fake_async_rollout.stop_global_worker = lambda: None
        path = Path(__file__).parents[1] / "rl" / "persistent_data_source.py"
        spec = importlib.util.spec_from_file_location("_test_persistent_data_source", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "torch": FakeTorch("torch"),
            "miles": types.ModuleType("miles"),
            "miles.rollout": types.ModuleType("miles.rollout"),
            "miles.rollout.base_types": SimpleNamespace(RolloutFnTrainOutput=lambda **kw: SimpleNamespace(**kw)),
            "miles.rollout.data_source": fake_data_source,
            "rl.fully_async_rollout": fake_async_rollout,
        }):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(
                    save=directory,
                    load=directory,
                    prover_async_no_progress_timeout_sec=1,
                    rollout_max_prompt_len=65535,
                )
                source = module.PersistentRolloutDataSource(args)
                self.assertIsNone(source.max_prompt_len_during_init)
                self.assertEqual(args.rollout_max_prompt_len, 65535)
                source.buffer.append(["retry"])
                source.record_completed(1, ["trained"])
                source.record_accepted(0, 1)
                source.record_completed(2, ["next-rollout"])
                source.record_accepted(1, 2)
                source.record_completed(3, ["surplus"])
                source.save(0)

                restored = module.PersistentRolloutDataSource(args)
                restored.load(0)
                self.assertEqual(
                    restored.buffer,
                    [["retry"], ["next-rollout"], ["surplus"]],
                )

                reset_args = SimpleNamespace(
                    save=directory,
                    load=directory,
                    prover_async_no_progress_timeout_sec=1,
                    rollout_max_prompt_len=65535,
                    prover_reset_rollout_data_state=True,
                )
                reset = module.PersistentRolloutDataSource(reset_args)
                reset.buffer.append(["must-not-cross-shards"])
                reset.load(0)
                self.assertEqual(reset.buffer, [])

    def test_mcp_bridge_times_out_and_releases_process(self):
        from agents.container import mcp_bridge

        proc = subprocess.Popen(
            ["sleep", "5"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        session = mcp_bridge.McpSession.__new__(mcp_bridge.McpSession)
        session.proc = proc
        session.next_id = 0
        try:
            with patch.object(mcp_bridge, "CALL_TIMEOUT", 0.05):
                with self.assertRaisesRegex(TimeoutError, "timed out"):
                    session._rpc("tools/call", {})
        finally:
            proc.kill()
            proc.wait(timeout=2)
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.close()
            proc.stdout.close()

    def test_launcher_uses_official_async_entrypoint_and_no_radix_cache_disable(self):
        launcher = (Path(__file__).parents[1] / "rl" / "run_prover_rl.sh").read_text()
        self.assertIn("train_async.py", launcher)
        self.assertIn("--deterministic-mode", launcher)
        self.assertNotIn("--async-save", launcher)
        self.assertIn("--no-save-optim", launcher)
        self.assertIn('"NCCL_ALGO"', launcher)
        self.assertIn('"CUBLAS_WORKSPACE_CONFIG"', launcher)
        self.assertIn("rl.fully_async_rollout.generate_rollout_fully_async", launcher)
        self.assertIn("rl.generate_with_prover.generate", launcher)
        self.assertIn("rl.persistent_data_source.PersistentRolloutDataSource", launcher)
        self.assertNotIn("--colocate", launcher)
        self.assertNotIn("disable-radix", launcher.lower())
        self.assertNotIn("logprob_start_len", launcher)
        self.assertIn("2:4)", launcher)
        self.assertIn("4:4)", launcher)
        self.assertIn("6:4)", launcher)
        self.assertIn("16:4)", launcher)
        self.assertIn("ACTOR_GPUS=4", launcher)
        self.assertIn("ROLLOUT_GPUS=4", launcher)
        self.assertIn("ROLLOUT_GPUS_PER_ENGINE=1", launcher)
        self.assertIn("SGLANG_EXPERT_PARALLEL=1", launcher)
        self.assertIn("ROLLOUT_GPUS=12", launcher)
        self.assertIn("ROLLOUT_GPUS=16", launcher)
        self.assertIn("ACTOR_NODES=2", launcher)
        self.assertIn("ACTOR_NODES=8", launcher)
        self.assertIn("ACTOR_GPUS=32", launcher)
        self.assertIn("ROLLOUT_GPUS=32", launcher)
        self.assertIn('--actor-num-nodes "${ACTOR_NODES}"', launcher)
        self.assertIn('--actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"', launcher)
        self.assertIn('--megatron-lm-root "${MEGATRON_LM_ROOT}"', launcher)
        self.assertIn('--load "${LOAD_CHECKPOINT_DIR}"', launcher)
        self.assertIn('LOAD_OPTIM_ARGS+=(--no-load-optim)', launcher)
        self.assertIn('SAVE_OPTIM_ARGS+=(--no-save-optim)', launcher)
        self.assertIn('--prover-reset-rollout-data-state', launcher)
        self.assertIn(
            '--optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION}"',
            launcher,
        )
        self.assertIn('OPTIMIZER_OFFLOAD_ARGS=()', launcher)
        self.assertIn('"${OPTIMIZER_OFFLOAD_ARGS[@]}"', launcher)
        self.assertIn('WEIGHT_CHECK_ARGS=()', launcher)
        self.assertIn('--check-weight-update-equal', launcher)
        self.assertIn('"${WEIGHT_CHECK_ARGS[@]}"', launcher)
        self.assertIn('--rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"', launcher)
        self.assertIn('--sglang-ep-size "${SGLANG_EXPERT_PARALLEL}"', launcher)
        self.assertIn('--over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"', launcher)
        self.assertIn('--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"', launcher)
        self.assertIn('--prover-max-total-tokens "${PROVER_MAX_TOTAL_TOKENS}"', launcher)
        self.assertIn('if (( MAX_TOKENS_PER_GPU < 1 )); then', launcher)
        self.assertIn('if (( LOG_PROBS_CHUNK_SIZE < 1 )); then', launcher)
        self.assertNotIn(
            'MAX_TOKENS_PER_GPU < PROVER_MAX_TOTAL_TOKENS',
            launcher,
        )
        self.assertIn(
            '--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}"',
            launcher,
        )
        self.assertIn('--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"', launcher)
        self.assertIn('PYTORCH_CUDA_ALLOC_CONF', launcher)
        self.assertIn('if [[ -z "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then', launcher)
        self.assertIn('--prover-max-tokens-per-turn "${PROVER_MAX_TOKENS_PER_TURN}"', launcher)
        self.assertIn(
            '--prover-max-truncation-nudges "${PROVER_MAX_TRUNCATION_NUDGES}"',
            launcher,
        )
        self.assertIn(
            '--prover-max-tool-result-tokens "${PROVER_MAX_TOOL_RESULT_TOKENS}"',
            launcher,
        )
        self.assertIn('--prover-router-timeout-sec "${PROVER_ROUTER_TIMEOUT_SEC:-180}"', launcher)
        self.assertIn(
            '--router-balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD}"',
            launcher,
        )
        self.assertIn("--sglang-moe-runner-backend flashinfer_cutlass", launcher)
        self.assertIn("--sglang-attention-backend trtllm_mha", launcher)


if __name__ == "__main__":
    unittest.main()
