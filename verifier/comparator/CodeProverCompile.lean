/- Trusted elaboration support for Lean 4.28.
   Give protected declarations a deterministic compiler caches, then restore the
   editable context so subsequent proof tactics can reuse their implementation's
   helpers. All declarations still undergo unchanged comparison and kernel checks. -/
module
public import Lean
import all Lean.Meta.Match.Match
import Lean.Meta.Tactic.AuxLemma

namespace CodeProver

private meta initialize savedMatcherExt : Lean.EnvExtension
    (Option (Lean.PHashMap Lean.Meta.Match.MatcherKey Lean.Name)) ←
  Lean.registerEnvExtension (pure none) (asyncMode := .local)

private meta initialize savedAuxLemmaExt : Lean.EnvExtension
    (Option Lean.Meta.AuxLemmas) ←
  Lean.registerEnvExtension (pure none) (asyncMode := .local)

public meta def beginSpecification : Lean.Elab.Command.CommandElabM Unit := do
  Lean.modifyEnv fun env =>
    let saved := Lean.Meta.Match.matcherExt.getState env
    let env := savedMatcherExt.setState env (some saved)
    let env := Lean.Meta.Match.matcherExt.setState env {}
    let saved := Lean.Meta.auxLemmasExt.getState env
    let env := savedAuxLemmaExt.setState env (some saved)
    Lean.Meta.auxLemmasExt.setState env {}

public meta def endSpecification : Lean.Elab.Command.CommandElabM Unit := do
  Lean.modifyEnv fun env =>
    let env := match savedMatcherExt.getState env with
      | none => env
      | some saved =>
        let merged := (Lean.Meta.Match.matcherExt.getState env).foldl
          (fun cache key name => if cache.contains key then cache else cache.insert key name) saved
        let env := savedMatcherExt.setState env none
        Lean.Meta.Match.matcherExt.setState env merged
    match savedAuxLemmaExt.getState env with
    | none => env
    | some saved =>
      let merged := (Lean.Meta.auxLemmasExt.getState env).lemmas.foldl
        (fun cache key value => if cache.contains key then cache else cache.insert key value) saved.lemmas
      let env := savedAuxLemmaExt.setState env none
      Lean.Meta.auxLemmasExt.setState env { lemmas := merged }

end CodeProver
