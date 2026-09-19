# Comparator integration (Lean/Mathlib 4.28)

Comparator is the **default, authoritative judge** for the V3 token-loop RL and
`rl.evaluate` pipeline. It assigns reward 1 only to an accepted proof. Configure
one backend (local image or CPU queue) and durable proof storage before launching.
There is no automatic fallback to the in-sandbox grader.

Use explicit `--prover-judge-mode shadow` for a paired comparison using the same
generated source, or `--prover-judge-mode legacy` to reproduce historical runs.
Frozen task folders, their Harbor graders, and published scores retain their
original contracts; the separate Harbor Trial adapter still uses those graders.

## Trusted implementation and pins

- [Comparator](https://github.com/leanprover/comparator/tree/d03acab154d269c06e60e4de7e4cc85deebff94b):
  `d03acab154d269c06e60e4de7e4cc85deebff94b` (`Comparator` library).
- Lean and Mathlib: **4.28.0**.
- lean4export base: `048394e1afeeb52b0fa27bcf3f1ade2ff0f0ab6d`.
  `export-4.28.patch` contains the upstream `Export.lean`/`Export/Parse.lean`
  changes through `076e8e57707e813375e8f9da8bf989799ace9680`: reject duplicate
  declarations, export the complete quotient package, handle missing/partial declarations.
- Lean4Checker replay: `b7398199245524275543dec6113229c9bb4902e5` (the dependency
  used by upstream Comparator's 4.28 tag).
- Base image digest is pinned in `Dockerfile`. Each invocation resolves the
  comparator image to an immutable local image ID and records it in its result.

The upstream 4.28 Comparator tag predates definition-hole support and later fixes.
We therefore build its current comparison/axiom traversal for 4.28 and provide
`CodeProverMain.lean`, an **offline adapter**, instead of upgrading the benchmark
or running an old comparator executable. The adapter replays into the 4.28 Lean
kernel and performs upstream's quotient post-check and primitive comparison.
This compatibility adapter is project-maintained, not an upstream release.

## Isolation and task contract

Three fresh Docker containers are created sequentially for each candidate:

1. **Challenge:** compile the trusted task-author baseline, discover theorem/definition
   targets from the elaborated environment, and export their declarations.
2. **Solution:** compile only the captured candidate source against the baked
   read-only Mathlib cache; export the trusted target list. This container never
   receives the challenge source/export. Its entire output is considered untrusted.
3. **Comparison:** read only textual exports and the trusted target list; compare
   declarations, check the axiom closure, and replay in a clean Lean kernel.
   Candidate `.olean` files are never loaded into this container.

All phases use the same module name (`Main`) in separate containers, so private
and namespaced declarations retain stable names. Targets come from the trusted
compiled challenge, using Lean declaration source ranges to exclude generated lemmas
and theorems in editable auxiliary sections. Candidates cannot supply the target list.
Definition holes are
original definitions whose bodies contain `sorryAx`. Non-hole dependencies,
preconditions, postconditions, imports, markers and signatures must stay intact.
The source check uses **byte-exact read-only projections**, including string
contents. Markerless or malformed original tasks require normalization before use.
Only changes to trailing EOF CR/LF counts are tolerated outside editable holes;
internal whitespace, strings, statements and all other protected bytes remain
exact. Original and candidate source bytes are retained without modification.

The trusted entrypoint imports `CodeProverCompile` and scopes Lean's matcher and auxiliary lemma caches
around the protected precondition/postcondition regions. This prevents an earlier
editable function from changing the generated matcher or proof names used by the fixed
specification. It restores earlier cache entries before editable code/proof regions,
so proof tactics can still reuse implementation helpers. Both sides receive the
same instrumentation. This is pinned Lean 4.28 compiler integration; upstream
`compareAt`, axiom validation, and independent kernel replay are unchanged.

Some unfinished templates contain auxiliary examples that cannot compile while
the implementation is still a hole. A task author may provide `challenge_source`
in the trusted queue catalog, or `--challenge` for the direct backend. The backend
requires its protected bytes to match the original exactly, saves `challenge.lean`,
and records its hash and override mode. Queue clients cannot choose this override.
Overrides require task-author review; the backend does not automatically erase helpers.
Preflight every trusted challenge before starting model generation.
Permitted axioms remain `propext`, `Quot.sound`, `Classical.choice`.

Each container is non-root, has a read-only root filesystem, no network or host
mounts, private `/work`/`input`/`tmp`, and limits of 4 CPUs, 8 GiB RAM and 256 PIDs.
Before Lean runs, the trusted entrypoint sets and verifies `no_new_privs` and
verifies that effective, permitted, inheritable and ambient capabilities are zero.
DSW rejects `--security-opt` and `--cap-drop`, so these restrictions are enforced
and checked in the entrypoint rather than changing the platform's Docker policy.
Candidate compilation therefore does not rely on Landlock or fake-landrun.
The host kernel, Docker isolation, trusted image/toolchain and Lean kernel remain
part of the trusted computing base. Extra independent kernels are not enabled.

## Build

On a machine with a **4.28.0** `lean` and `lake` already on `PATH`:

```sh
bash verifier/comparator/build.sh /tmp/comparator-build
bash verifier/comparator/build-image.sh /tmp/comparator-build codeprover-comparator:lean4.28-d03acab
```

The first directory must be empty. Source revisions and the base image are pinned;
the scripts do not change the host's default Lean version or publish any image.
Rebuild the image after changing the Lean adapter or `phase.py`.

## Use in rollout/evaluation

Add to the existing prover arguments:

```sh
--prover-judge-mode comparator \
--prover-comparator-image codeprover-comparator:lean4.28-d03acab \
--prover-proof-artifacts-dir /durable/run/path/proofs \
--prover-comparator-timeout-sec 1200 \
--prover-comparator-concurrency 2
```

Concurrency is per rollout process; plan CPU/RAM for the aggregate across workers.
A shadow run keeps the legacy reward and records the comparator verdict and any
disagreement. Compiler child exit codes, elapsed time, peak RSS, and available cgroup memory
counters are recorded for diagnosing runtime failures. A candidate compiler killed
by its container's OOM controller is a `solution_memory_limit` rejection under the
fixed verification budget. This classification requires SIGKILL plus an increased
kernel `oom_kill` counter; an unexplained kill remains an infrastructure error.
Trusted challenge failures are always infrastructure errors. Memory-limit rejection
means the answer could not be verified within budget, not a mathematical counterexample.
Malformed challenge or solution exports are infrastructure errors: parsing must
finish before the comparator can issue a mathematical verdict.
Unavailable comparator results have `comparator_available=0` and
no `comparator_accepted` score; they are not counted as proof failures.
Authoritative `comparator` mode skips the old judge and propagates infrastructure
failures to the rollout's existing failure/retry mechanism, with no fallback.
Cancellation always propagates and triggers container cleanup.

For direct mode, the rollout worker needs a Docker daemon and comparator image.
For **DLC/E2B workers without Docker**, use the shared-filesystem queue transport:

```sh
# On the CPU verifier node, create a trusted catalog once and start the service.
python3 -m verifier.comparator.queue catalog --task-root /trusted/tasks --output /run/catalog.json
python3 -m verifier.comparator.queue serve --queue /shared/run/queue \
  --catalog /run/catalog.json --image codeprover-comparator:lean4.28-d03acab \
  --concurrency 4 --timeout 1200

# On rollout workers, use this instead of --prover-comparator-image:
--prover-comparator-queue-dir /shared/run/queue
```

Keep the durable proof artifact directory. Omit `--prover-judge-mode` to use
authoritative Comparator, or choose `shadow` explicitly for a paired audit.
`--prover-comparator-timeout-sec` includes queue waiting in this mode; the service
has its own per-proof timeout. Requests are published atomically and include
source/catalog hashes. Original source is resolved from the CPU node's trusted
catalog, never supplied by a candidate. Results are matched by request/source
identity. Infrastructure retries reuse the exact retained proof bytes. Heartbeat
failures, timeouts and cancellation are explicit; no network listener is opened.
Only trusted rollout/controller processes may access this queue, and agent
sandboxes must not mount it. A kernel-owned lock prevents duplicate dispatchers.
The worker supports restart/reclaim after a crash. Stopped/expired requests do not
silently become valid scores.

Directory discovery runs off the asynchronous event loop. A separate heartbeat
continues while shared storage is slow, and completed requests are cached to
avoid probing every old result on each scan. Heartbeat write failures stop the
service; shutdown waits for an in-flight heartbeat before publishing `stopped`.
The client freshness limit and request identity checks remain unchanged.

Queue reads retry transient `ESTALE` up to six total attempts (1.55 seconds of
backoff). `EIO` is retried only after an observed `ESTALE`, including when close
masks the original read exception. Standalone `EIO`, storage-full and permission
errors propagate immediately.

### Launchers and preflight

`rl/submit_dlc.py` accepts the same six `--prover-*` judge flags. Its default proof
path is `<run-root>/<run-id>/proofs`; queue and output paths must be visible on the
DLC data mount. The submitter forwards judge settings to `rl/run_prover_rl.sh`,
which also accepts the corresponding environment variables:

```sh
export PROVER_JUDGE_MODE=comparator
export PROVER_COMPARATOR_QUEUE_DIR=/shared/run/queue
export PROVER_PROOF_ARTIFACTS_DIR=/shared/run/proofs
export PROVER_COMPARATOR_TIMEOUT_SEC=1200
export PROVER_COMPARATOR_CONCURRENCY=2
```

For a local Docker backend, set `PROVER_COMPARATOR_IMAGE` and leave the queue
unset. Configuring both backends is an error. Preflight checks a fresh service
heartbeat or the local image, and writes a temporary probe to the proof directory
before loading models. Training-only replay skips judge checks. Episode and
no-progress deadlines include a separate Comparator reserve.

`rl/local_episode_test.py` accepts the same flags and runs the same preflight.

Every Comparator/shadow episode captures its source before any legacy verification. A UUID
artifact directory stores `original.lean`, `solution.lean`, `judging.json`, and a
`comparator/` directory with targets, source hash, image ID, verdict, timing and
bounded logs. In queue mode, backend artifacts live under the queue request;
`judging.json` records that location and the request ID. Authoritative infrastructure
errors are recorded with `accepted: null` before propagating. Evaluation JSONL includes `metadata.judge_details` and
`metadata.rewards`. Keep the artifact directory on durable storage. Past runs
without retained final source cannot automatically be rejudged offline.

For an already-saved solution (no model/GPU needed):

```sh
python3 -m verifier.comparator.backend \
  --original path/to/original.lean --solution path/to/solution.lean \
  --image codeprover-comparator:lean4.28-d03acab \
  --artifacts /durable/run/path/new-attempt
```

## Verification

```sh
python3 -m unittest discover -s tests -p test_judge_config.py
python3 -m unittest discover -s tests -p test_comparator_queue.py
CODEPROVER_COMPARATOR_TEST_IMAGE=codeprover-comparator:lean4.28-d03acab \
  python3 -m unittest discover -s tests -p test_comparator.py
```

The real suite covers Mathlib oracle positives, namespaced/private declarations,
Verina-style function/specification pairs, no-op and invalid proofs, an answer
that redefines `False` through notation while still compiling, a direct
`sorryAx` dependency, and a candidate attempting to write the immutable project.
Additional runtime tests cover editable helper removal, matcher reuse across
protected declarations, and reviewed challenge overrides.
Unit tests cover strict source integrity, source persistence, shadow disagreement,
infrastructure handling and cancellation. No generated dataset is changed.

## Attribution

`CodeProverMain.lean` adapts upstream Comparator's executable for offline 4.28
replay. `export-4.28.patch` backports the lean4export fixes at the revisions above.
Both upstream projects use the Apache 2.0 license, included as [LICENSE](LICENSE).
The build fetches the pinned libraries; it does not vendor their full source tree.
