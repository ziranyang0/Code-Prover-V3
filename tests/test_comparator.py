from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from verifier.comparator.backend import ComparatorInfrastructureError, judge, source_checks
from rl.comparator_judge import grade_with_comparator

MATH = '''import Mathlib
namespace Regression
theorem math_ok (n : Nat) : n + 0 = n := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
end Regression
'''
CODE = '''import Mathlib
namespace Regression
def increment (n : Nat) : Nat :=
  -- !benchmark @start code
  sorry
  -- !benchmark @end code
def post (n r : Nat) : Prop := r = n + 1
theorem increment_spec (n : Nat) : post n (increment n) := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
end Regression
'''

class SourceTests(unittest.TestCase):
    def test_string_whitespace_semantic_change_is_rejected(self):
        original = MATH.replace('n + 0 = n', '("a b" : String) = "a  b"')
        final = original.replace('"a  b"', '"a b"').replace('  sorry', '  rfl')
        self.assertFalse(source_checks(original, final)['spec_intact'])

    def test_eof_newline_changes_are_allowed_without_changing_source(self):
        original = MATH + '\n\n'
        final = MATH.replace('  sorry', '  simp').rstrip('\n')
        checks = source_checks(original, final)
        self.assertTrue(checks['spec_intact'])
        self.assertEqual(checks['spec_detail']['mode'], 'byte_exact_except_eof_newlines')
        self.assertTrue(source_checks(MATH, MATH.replace('  sorry', '  simp') + '\n\n')['spec_intact'])

    def test_eof_tolerance_does_not_allow_internal_whitespace_changes(self):
        original = MATH + '\n\n'
        final = MATH.replace('namespace Regression', 'namespace  Regression').replace('  sorry', '  simp')
        self.assertFalse(source_checks(original, final)['spec_intact'])
        final = MATH.replace('namespace Regression', 'namespace Regression\n').replace('  sorry', '  simp')
        self.assertFalse(source_checks(original, final)['spec_intact'])

    def test_eof_tolerance_does_not_hide_changed_string_or_statement(self):
        original = MATH.replace('n + 0 = n', '("a b" : String) = "a  b"') + '\n\n'
        final = original.replace('"a  b"', '"a b"').replace('  sorry', '  rfl').rstrip('\n')
        self.assertFalse(source_checks(original, final)['spec_intact'])
        self.assertFalse(source_checks(MATH+'\n', MATH.replace('n + 0 = n', 'True').rstrip('\n'))['spec_intact'])

    def test_changed_postcondition_is_rejected(self):
        self.assertFalse(source_checks(CODE, CODE.replace('r = n + 1', 'True'))['spec_intact'])

    def test_malformed_trusted_markers_are_infrastructure(self):
        original = MATH.replace('-- !benchmark @end proof', '-- !benchmark @end code')
        with self.assertRaises(ComparatorInfrastructureError):
            source_checks(original, original)

    def test_markers_required(self):
        with self.assertRaises(ComparatorInfrastructureError):
            source_checks('theorem t : True := by sorry', 'theorem t : True := by trivial')

    def test_empty_and_noop_answers(self):
        self.assertFalse(source_checks(MATH, '')['spec_intact'])
        self.assertFalse(source_checks(MATH, MATH)['sorry_free'])

    def test_editable_code_allowed(self):
        final = CODE.replace('  sorry', '  n + 1', 1).replace('  sorry', '  rfl')
        checks = source_checks(CODE, final)
        self.assertTrue(all(checks[k] for k in ('spec_intact', 'sorry_free', 'forbidden_free')))

class WiringTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'original.lean').write_text(MATH)
        (self.root / 'task_file.txt').write_text('/task/proof.lean')
        self.cfg = SimpleNamespace(judge_mode='shadow', comparator_image='test-image',
                                   proof_artifacts_dir=str(self.root / 'artifacts'),
                                   comparator_timeout_sec=60, comparator_concurrency=2)
        self.sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(
            return_code=0, stdout=MATH.replace('  sorry', '  simp'))))
        self.legacy = AsyncMock(return_value=(1.0, {'reward': 1.0}))

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def test_shadow_preserves_legacy_and_records_disagreement_and_source(self):
        with patch('rl.comparator_judge.judge', AsyncMock(return_value={'accepted':False, 'status':'rejected'})):
            reward, metrics, details = await grade_with_comparator(self.sandbox, self.root, self.cfg, self.legacy)
        self.assertEqual(reward, 1)
        self.assertEqual(metrics['comparator_accepted'], 0)
        self.assertTrue(details['disagreement'])
        self.assertEqual((Path(details['artifact_dir'])/'solution.lean').read_text(), self.sandbox.exec.return_value.stdout)

    async def test_shadow_unavailable_is_not_scored_as_rejected(self):
        with patch('rl.comparator_judge.judge', AsyncMock(side_effect=ComparatorInfrastructureError('image missing'))):
            reward, metrics, details = await grade_with_comparator(self.sandbox, self.root, self.cfg, self.legacy)
        self.assertEqual(reward, 1)
        self.assertEqual(metrics['comparator_available'], 0)
        self.assertNotIn('comparator_accepted', metrics)
        self.assertIsNone(details['disagreement'])

    async def test_authoritative_mode_does_not_fall_back_on_infra_error(self):
        self.cfg.judge_mode = 'comparator'
        with patch('rl.comparator_judge.judge', AsyncMock(side_effect=ComparatorInfrastructureError('broken'))):
            with self.assertRaises(ComparatorInfrastructureError):
                await grade_with_comparator(self.sandbox, self.root, self.cfg, self.legacy)
        self.legacy.assert_not_awaited()
        audit = json.loads(next((self.root / 'artifacts').glob('*/judging.json')).read_text())
        self.assertEqual(audit['comparator']['status'], 'infrastructure_error')
        self.assertIsNone(audit['comparator']['accepted'])
        self.assertTrue((Path(audit['artifact_dir']) / 'solution.lean').is_file())

    async def test_authoritative_acceptance_and_invalid_verdict(self):
        self.cfg.judge_mode = 'comparator'
        with patch('rl.comparator_judge.judge', AsyncMock(return_value={'accepted': True})):
            reward, _, _ = await grade_with_comparator(self.sandbox, self.root, self.cfg, self.legacy)
        self.assertEqual(reward, 1)
        with patch('rl.comparator_judge.judge', AsyncMock(return_value={'accepted': None})):
            with self.assertRaises(ComparatorInfrastructureError):
                await grade_with_comparator(self.sandbox, self.root, self.cfg, self.legacy)
        self.legacy.assert_not_awaited()

    async def test_authoritative_rejection_sets_reward_zero(self):
        self.cfg.judge_mode = 'comparator'
        with patch('rl.comparator_judge.judge', AsyncMock(return_value={'accepted':False, 'status':'rejected'})):
            reward, _, _ = await grade_with_comparator(self.sandbox, self.root, self.cfg, self.legacy)
        self.assertEqual(reward, 0)
        self.legacy.assert_not_awaited()

    async def test_cancellation_propagates_from_shadow(self):
        with patch('rl.comparator_judge.judge', AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await grade_with_comparator(self.sandbox, self.root, self.cfg, self.legacy)

@unittest.skipUnless(os.environ.get('CODEPROVER_COMPARATOR_TEST_IMAGE'), 'set image to enable real Docker/Lean smoke')
class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def check(self, original, solution):
        with tempfile.TemporaryDirectory() as tmp:
            result = await judge(original, solution, image=os.environ['CODEPROVER_COMPARATOR_TEST_IMAGE'],
                                 artifacts=Path(tmp)/'artifacts', timeout=600)
            # Print useful evidence without large Lean exports.
            print(json.dumps({'accepted':result['accepted'], 'reason':result['reason'],
                              'targets':result.get('targets'), 'seconds':result['duration_sec']}), flush=True)
            return result

    async def test_editable_auxiliary_theorem_is_not_a_task_target(self):
        original = MATH.replace('namespace Regression',
            'namespace Regression\n-- !benchmark @start proof_aux\n'
            'theorem optional_helper : True := by trivial\n-- !benchmark @end proof_aux')
        final = original.replace('theorem optional_helper : True := by trivial', '').replace('  sorry', '  simp')
        result = await self.check(original, final)
        self.assertTrue(result['accepted'])
        self.assertNotIn('Regression.optional_helper', result['targets']['theorem_names'])

    async def test_editable_matcher_does_not_change_protected_postcondition(self):
        original = """import Mathlib
-- !benchmark @start code_aux
-- !benchmark @end code_aux
def f (xs : List Nat) : Nat :=
  -- !benchmark @start code
  sorry
  -- !benchmark @end code
-- !benchmark @start postcond_aux
-- !benchmark @end postcond_aux
def post (xs : List Nat) (r : Nat) : Prop :=
  r = xs.foldl (fun a x => match a with | 0 => x | n+1 => n+1+x) 0
-- !benchmark @start proof_aux
-- !benchmark @end proof_aux
theorem f_spec (xs : List Nat) : post xs (f xs) := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
"""
        final = original.replace('-- !benchmark @start code_aux',
            '-- !benchmark @start code_aux\ndef helper (xs : List Nat) : Nat := '
            'xs.foldl (fun a x => match a with | 0 => x | n+1 => n+1+x) 0')
        final = final.replace('  sorry', '  helper xs', 1).replace('  sorry', '  rfl')
        self.assertTrue((await self.check(original, final))['accepted'])

    async def test_proof_can_reuse_implementation_matcher_after_protected_spec(self):
        original = """import Mathlib
-- !benchmark @start code_aux
-- !benchmark @end code_aux
def select (x : Option Nat) : Nat :=
  -- !benchmark @start code
  sorry
  -- !benchmark @end code
-- !benchmark @start postcond_aux
-- !benchmark @end postcond_aux
def selected (r n : Nat) : Prop := r = n
-- !benchmark @start proof_aux
-- !benchmark @end proof_aux
theorem select_spec (x : Option Nat) : selected (select x) (x.getD 0) := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
"""
        final = original.replace('  sorry', '  match x with | some a => a | none => 0', 1)
        final = final.replace('  sorry', """  unfold select
  have hres : (match x with | some a => a | none => 0) = x.getD 0 := by cases x <;> rfl
  rw [hres]
  rfl""")
        self.assertTrue((await self.check(original, final))['accepted'])

    async def test_implementation_auxiliary_proof_does_not_change_protected_spec(self):
        original = """import Mathlib
-- !benchmark @start code_aux
-- !benchmark @end code_aux
def ordered (xs : Array Int) : Bool :=
  -- !benchmark @start code
  sorry
  -- !benchmark @end code
-- !benchmark @start postcond_aux
-- !benchmark @end postcond_aux
def ordered_post (xs : Array Int) (result : Bool) : Prop :=
  (∀ i, (hi : i < xs.size - 1) → xs[i] ≤ xs[i + 1]) ↔ result
-- !benchmark @start proof_aux
-- !benchmark @end proof_aux
theorem ordered_spec (xs : Array Int) : ordered_post xs (ordered xs) := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
"""
        final = original.replace('  sorry',
            '  decide (∀ i, (hi : i < xs.size - 1) → xs[i] ≤ xs[i + 1])', 1)
        final = final.replace('  sorry', '  exact decide_eq_true_iff.symm')
        self.assertTrue((await self.check(original, final))['accepted'])

    async def test_corrupt_solution_export_is_infrastructure(self):
        from verifier.comparator import backend
        container = backend._container
        async def corrupt_export(image, phase, inputs, outputs, **kwargs):
            code = await container(image, phase, inputs, outputs, **kwargs)
            if phase == 'solution' and code == 0:
                # Model an exporter/transport failure after an otherwise valid proof.
                with (outputs / 'solution.ndjson').open('a') as stream:
                    stream.write('not a JSON object\n')
            return code
        with patch.object(backend, '_container', corrupt_export):
            with self.assertRaisesRegex(ComparatorInfrastructureError, 'comparator runtime failed'):
                await self.check(MATH, MATH.replace('  sorry', '  simp'))

    async def test_task_author_challenge_repairs_only_editable_template(self):
        original = MATH.replace('namespace Regression',
            '-- !benchmark @start code_aux\ndef unused := missingTemplateHelper\n'
            '-- !benchmark @end code_aux\nnamespace Regression')
        challenge = original.replace('def unused := missingTemplateHelper', '')
        final = challenge.replace('  sorry', '  simp')
        with tempfile.TemporaryDirectory() as folder:
            result = await judge(original, final, challenge=challenge,
                image=os.environ['CODEPROVER_COMPARATOR_TEST_IMAGE'], artifacts=Path(folder)/'proof')
            self.assertTrue(result['accepted'])
            self.assertEqual((Path(folder)/'proof/original.lean').read_text(), original)
            self.assertEqual((Path(folder)/'proof/challenge.lean').read_text(), challenge)
            self.assertEqual(result['challenge_mode'], 'trusted_override')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ComparatorInfrastructureError, 'protected task bytes'):
                await judge(original, final, challenge=challenge.replace('n + 0 = n', 'True'),
                    image=os.environ['CODEPROVER_COMPARATOR_TEST_IMAGE'], artifacts=Path(folder)/'proof')

    async def test_candidate_oom_is_distinguished_from_challenge_failure(self):
        from verifier.comparator import backend
        original = """import Lean
-- !benchmark @start proof_aux
-- !benchmark @end proof_aux
theorem small : True := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
"""
        final = original.replace('-- !benchmark @start proof_aux',
            '-- !benchmark @start proof_aux\n#eval IO.println (List.range 100000000).length').replace('  sorry', '  trivial')
        command = backend._command
        async def bounded_command(args, **kwargs):
            args = ['--memory=512m' if a == '--memory=8g' else a for a in args]
            return await command(args, **kwargs)
        with patch.object(backend, '_command', bounded_command):
            result = await self.check(original, final)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['reason'], 'solution_memory_limit')
            with self.assertRaisesRegex(ComparatorInfrastructureError, 'trusted challenge failed'):
                await self.check(final, final)

    async def test_unexplained_candidate_kill_remains_infrastructure(self):
        original = MATH.replace('namespace Regression',
            '-- !benchmark @start proof_aux\n-- !benchmark @end proof_aux\nnamespace Regression')
        final = original.replace('-- !benchmark @start proof_aux',
            '-- !benchmark @start proof_aux\n#eval IO.Process.run {cmd := "/bin/sh", args := #["-c", "kill -KILL $PPID"]}').replace('  sorry', '  simp')
        with self.assertRaisesRegex(ComparatorInfrastructureError, 'solution runtime failed'):
            await self.check(original, final)

    async def test_math_oracle_positive(self):
        result = await self.check(MATH, MATH.replace('  sorry', '  simp'))
        self.assertTrue(result['accepted'])
        self.assertIn('Regression.math_ok', result['targets']['theorem_names'])

    async def test_verina_code_oracle_positive(self):
        result = await self.check(CODE, CODE.replace('  sorry', '  n + 1', 1).replace('  sorry', '  rfl'))
        self.assertTrue(result['accepted'])
        self.assertIn('Regression.increment', result['targets']['definition_names'])

    async def test_eof_only_formatting_still_reaches_kernel(self):
        original = MATH + '\n\n'
        final = MATH.replace('  sorry', '  simp').rstrip('\n')
        result = await self.check(original, final)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['reason'], 'comparator')
        self.assertFalse((await self.check(original, MATH.rstrip('\n')))['accepted'])

    async def test_noop_negative(self):
        self.assertFalse((await self.check(MATH, MATH))['accepted'])

    async def test_wrong_proof_negative(self):
        self.assertFalse((await self.check(MATH, MATH.replace('  sorry', '  exact False.elim (by assumption)')))['accepted'])

    async def test_semantic_notation_attack_negative(self):
        original = '''import Mathlib
-- !benchmark @start proof_aux
-- !benchmark @end proof_aux
theorem victim : False := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
'''
        final = original.replace('-- !benchmark @start proof_aux',
                                 '-- !benchmark @start proof_aux\nlocal notation "False" => True').replace('  sorry', '  trivial')
        result = await self.check(original, final)
        self.assertFalse(result['accepted'])
        # This attack compiles. It must reach and be rejected by comparator.
        self.assertEqual(result['reason'], 'comparator')

    async def test_sorry_axiom_hidden_behind_macro_negative(self):
        original = '''import Mathlib
-- !benchmark @start proof_aux
-- !benchmark @end proof_aux
theorem victim : False := by
  -- !benchmark @start proof
  sorry
  -- !benchmark @end proof
'''
        # A direct sorryAx reference bypasses the old sorry-token scan. Comparator
        # must reject its axiom closure even though the Lean file compiles.
        final = original.replace('  sorry', '  exact sorryAx False true')
        result = await self.check(original, final)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'comparator')

    async def test_solution_cannot_modify_baked_project(self):
        original = MATH.replace('namespace Regression',
            '-- !benchmark @start proof_aux\n-- !benchmark @end proof_aux\nnamespace Regression')
        guard = '\n'.join([
            '#eval do',
            '  let writable ← try',
            '    IO.FS.writeFile "/task/lakefile.toml" "tampered"',
            '    pure true',
            '  catch _ => pure false',
            '  if writable then throw (IO.userError "trusted project was writable")'])
        solution = original.replace('-- !benchmark @start proof_aux',
                                    '-- !benchmark @start proof_aux\n' + guard).replace('  sorry', '  simp')
        self.assertTrue((await self.check(original, solution))['accepted'])

    async def test_private_target_positive(self):
        original = MATH.replace('theorem math_ok', 'private theorem math_ok')
        self.assertTrue((await self.check(original, original.replace('  sorry', '  simp')))['accepted'])
