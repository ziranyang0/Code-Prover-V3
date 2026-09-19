/-
Copyright (c) 2025 Lean FRO, LLC. All rights reserved.
Released under Apache 2.0 license as described in the upstream LICENSE.

Offline adapter for Lean 4.28. The comparison/axiom traversal is upstream
Comparator at d03acab154d269c06e60e4de7e4cc85deebff94b. Kernel replay uses the
4.28 Lean4Checker; quotient post-check and primitive list follow upstream Main.
Only textual exports enter this executable. It never imports candidate oleans.
-/
import Comparator
import Lean4Checker.Replay
import Export.Parse

open Lean

def primitiveTargets : Array Name := #[
  ``Nat.add, ``Nat.sub, ``Nat.mul, ``Nat.pow, ``Nat.gcd, ``Nat.div, ``Nat.mod,
  ``Nat.beq, ``Nat.ble, ``Nat.land, ``Nat.lor, ``Nat.xor, ``Nat.shiftLeft,
  ``Nat.shiftRight, ``String.ofList, ``Char.ofNat, ``List, ``eagerReduce,
  ``Nat, ``String, ``String.mk, ``Char, ``optParam, ``autoParam,
  ``semiOutParam, ``outParam]

def legalAxioms : Array Name := #[``propext, ``Quot.sound, ``Classical.choice]

def exportPrimitives : Array Name := primitiveTargets ++ legalAxioms ++
  #[``Quot, ``Quot.mk, ``Quot.lift, ``Quot.ind]

structure TargetConfig where
  theorem_names : Array String
  definition_names : Array String
  deriving FromJson, ToJson

def readExport (path : System.FilePath) : IO Export.ExportedEnv := do
  let handle ← IO.FS.Handle.mk path .read
  Export.parseStream (IO.FS.Stream.ofHandle handle)

def parseTarget (n : String) : IO Name := do
  let some name := Syntax.decodeNameLit ("`" ++ n)
    | throw <| .userError s!"Invalid target name: {n}"
  return name

def verify (cfg : TargetConfig) (challenge solution : Export.ExportedEnv) : IO Unit := do
  if cfg.theorem_names.isEmpty then
    throw <| .userError "At least one trusted theorem target is required"
  let theorems ← cfg.theorem_names.mapM parseTarget
  let definitions ← cfg.definition_names.mapM parseTarget
  IO.ofExcept <| Comparator.compareAt challenge solution (theorems ++ legalAxioms)
    definitions primitiveTargets
  IO.ofExcept <| Comparator.checkAxioms solution theorems definitions legalAxioms
  let env ← Lean.mkEmptyEnvironment
  let quotTargets := [`Quot.mk, `Quot.lift, `Quot.ind]
  let constants := quotTargets.foldl (init := solution.constMap) (·.erase ·)
  let env ← env.replay' constants
  for target in `Quot :: quotTargets do
    if let some info := solution.constMap[target]? then
      let some checked := env.toKernelEnv.find? target
        | throw <| .userError s!"Missing quotient constant after kernel replay: {target}"
      if info != checked then
        throw <| .userError s!"Quotient post-check mismatch: {target}"

-- Exit 1 is a mathematical rejection. Crashes/signals/timeouts remain infra errors.
def run (args : List String) : IO UInt32 := do
  if args == ["--primitives"] then
    IO.println <| Json.compress <| toJson (exportPrimitives.map Name.toString)
    return (0 : UInt32)
  let [configPath, challengePath, solutionPath] := args
    | throw <| .userError "Expected config.json challenge.ndjson solution.ndjson"
  let cfg ← IO.ofExcept <| fromJson? (← IO.ofExcept <| Json.parse (← IO.FS.readFile configPath))
  let challenge ← readExport challengePath
  -- Both exports are produced by the trusted exporter/transport pipeline.
  -- Unreadable exports provide no mathematical verdict, even on the solution side.
  let solution ← readExport solutionPath
  try
    verify cfg challenge solution
  catch e =>
    IO.eprintln e.toString
    return (1 : UInt32)
  IO.println "Comparator accepted the solution"
  return (0 : UInt32)

-- Trusted setup/parse errors have a different exit status from proof rejection.
def main (args : List String) : IO UInt32 := do
  try run args
  catch e =>
    IO.eprintln e.toString
    return (2 : UInt32)
