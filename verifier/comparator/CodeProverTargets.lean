/- Trusted challenge introspection. Explicit declarations are identified through
   Lean's source ranges; generated compiler lemmas and editable auxiliary proofs
   are not task targets. Names and types still come from the elaborated environment. -/
import Lean
open Lean

def editableAuxLines (source : String) : Array Nat := Id.run do
  let mut active := false
  let mut lines : Array Nat := #[]
  for (line, idx) in (source.splitOn "\n").zipIdx do
    let text := line.trimAscii.toString
    if ["solution_aux", "code_aux", "proof_aux"].any
        (fun name => text == "-- !benchmark @start " ++ name) then
      active := true
    else if text.startsWith "-- !benchmark @end " then
      active := false
    else if active then
      lines := lines.push (idx + 1)
  return lines

def main (_ : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `Main }] {}
  let some idx := env.getModuleIdx? `Main
    | throw <| .userError "Missing trusted Main module"
  let editable := editableAuxLines (← IO.FS.readFile "/work/Main.lean")
  let mut theorems : Array String := #[]
  let mut definitions : Array String := #[]
  for (name, info) in env.constants.toList do
    if env.getModuleIdxFor? name != some idx then continue
    let some ranges := declRangeExt.find? (level := .exported) env name <|>
        declRangeExt.find? (level := .server) env name | continue
    if editable.contains ranges.selectionRange.pos.line then continue
    match info with
    | .thmInfo _ => theorems := theorems.push name.toString
    | .defnInfo info =>
      if info.value.getUsedConstants.contains `sorryAx then
        definitions := definitions.push name.toString
    | _ => pure ()
  if theorems.isEmpty then
    throw <| .userError "Trusted task has no explicit theorem targets"
  IO.println <| Json.compress <| Json.mkObj [
    ("theorem_names", toJson (theorems.qsort (· < ·))),
    ("definition_names", toJson (definitions.qsort (· < ·)))]
