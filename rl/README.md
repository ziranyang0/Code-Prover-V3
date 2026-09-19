# Async RL with E2B-compatible sandboxes

The launcher supports the configured eight- or ten-worker GPU topology.
The example worker shape uses 8 GPUs, 192 CPUs, 1800 GiB RAM and 1800 GiB
shared memory. RDMA is enabled; the scheduler and model select the attention
backend. Confirm capacity with a GPU smoke test before training.

DLC names must start with `TRACES_Verification_RL_`. Reuse one `--run-id`
across stages and use distinct job names. Each example stage reads a
960-task shard and trains for 30 rollout steps with 32 prompts x 8 samples.
Dynamic filtering can revisit tasks, so steps do not guarantee a dataset pass.

## Deployment configuration

Supply deployment addresses, resource identifiers, image references and local
paths at runtime. These deployment identifiers have no built-in private defaults.
The submitter requires `--workspace-id`, `--resource-id`, `--data-source-id`
and an absolute `--mount-path`; the corresponding `DLC_WORKSPACE_ID`,
`DLC_RESOURCE_ID`, `DLC_DATA_SOURCE_ID` and `DLC_MOUNT_PATH` environment variables
can also supply them. Optional labels use `--topic-id` / `DLC_TOPIC_ID` and
`--owner` / `DLC_JOB_OWNER`.

## Default proof judge

Training rollouts and `rl.evaluate` use **Comparator** by default, with independent
Lean 4.28 kernel replay. Supply `--prover-comparator-queue-dir /shared/run/queue`
for a CPU verifier service, or `--prover-comparator-image IMAGE` for local Docker.
The DLC submitter stores proofs under `<run-root>/<run-id>/proofs` by default;
override with `--prover-proof-artifacts-dir /durable/proofs`. It checks that paths
are visible on the data mount. The launcher checks service/image availability and
writable storage before models load. Queue capacity and timeout must cover the
aggregate rollout traffic.

Comparator infrastructure failures remain failed episodes and never become a
normal zero reward or trigger fallback to the old judge. Evaluation JSONL retains
judge details and proof locations. Choose `--prover-judge-mode shadow` explicitly
for paired audits or `legacy` for historical reproduction. Existing Harbor task
graders retain their frozen behavior. See [Comparator setup, pins and tests](../verifier/comparator/README.md).

## Image

Build the public, digest-pinned Miles base from the Dockerfile:

```bash
docker build --platform linux/amd64 -t codeprover-rl:local rl/image
```

The image installs the pinned E2B/OpenAI runtime and checks dependencies.
Source code and checkpoints are mounted separately. Publish it to your own
registry and pass the resulting digest reference through `--image` or
`CODEPROVER_RL_IMAGE`. Submission requires a published registry URI pinned by
digest; no private registry address is built into the submitter.

## Self-hosted sandbox

Use your service's API origin, domain, credential file, CA bundle and verified
Lean template. Example hostnames below are reserved documentation names:

```bash
--e2b-api-url https://api.sandbox.example \
--e2b-domain sandbox.example \
--e2b-key-file /secure/path/sandbox.key \
--e2b-ca-bundle /secure/path/ca-bundle.pem \
--no-e2b-validate-api-key \
--e2b-template YOUR_VERIFIED_LEAN_TEMPLATE
```

Retain TLS verification and combine the private CA with existing public roots.
Disable SDK key-format validation only for services using another credential
format; server authentication remains required. For direct launch/evaluation,
set `E2B_API_URL`, `E2B_DOMAIN`, `E2B_API_KEY_FILE`, `SSL_CERT_FILE`,
`E2B_VALIDATE_API_KEY` and an explicit `E2B_TEMPLATE_ID` as appropriate.
Select a template containing Lean/Mathlib 4.28 and the prover skeleton.

OpenKruise deployments can provide `E2B_CREATE_METADATA_JSON` with their
`e2b.agents.kruise.io/image` digest reference and documented claim settings.
The `lean-base` bootstrap profile prepares task directory permissions and
reuses the precompiled dependency tree. Agent commands run as the sandbox
user. See the [OpenKruise claim extensions](https://openkruise.io/kruiseagents/user-manuals/sandbox-claim).

## Client-side Lean warm pool

The fully asynchronous rollout worker can keep fresh, initialized E2B sandboxes
ready before assigning episodes. A capacity-planning **candidate**, which must be measured on your service, uses:

```bash
--e2b-warm-pool-size 3072 \
--e2b-create-concurrency 512 \
--e2b-warm-pool-wait-timeout-sec 3600
```

[configs/aliyun-sg-warm-pool-candidate.sh](configs/aliyun-sg-warm-pool-candidate.sh)
contains example settings and requires routing/image values via environment
variables. It also supplies the 960-second create
and 1200-second async pause deadlines, and a separate 384-episode limit.
From `Code-Prover-V3`, source it before the submission command below and append
`"${ALIYUN_SG_WARM_POOL_ARGS[@]}" --e2b-ca-bundle /absolute/path/to/sg-ca-bundle.pem`.
Use the matching service key for that command's `--e2b-key-file`. Sourcing the file only defines
a Bash argument array; it does not allocate sandboxes or submit training.
GPU topology, token budgets, task paths and checkpoints remain explicit DLC
inputs. The candidate still needs a 384-way claim and sustained-refill trial.

Direct launcher/evaluation uses `E2B_WARM_POOL_SIZE`,
`E2B_CREATE_CONCURRENCY`, and `E2B_WARM_POOL_WAIT_TIMEOUT_SEC`. The default pool
size is zero. Keep the existing endpoint, CA, template, pinned image metadata
and `lean-base` bootstrap configuration. This pool uses the E2B API and does
not change the server's SandboxSet.

The worker pre-fills before fetching rollout groups. Each instance must pass
the ordinary user's Lean/Mathlib runtime check before becoming available.
A claim renews the sandbox for the complete episode budget, then the episode
uploads its task and destroys the instance when finished. Used instances
never return to the pool. Task attribution is recorded with `sandbox_id` in
episode metadata and the claim log; service metadata identifies the owning
run, attempt and pool before any task is assigned.

Pool size bounds idle, creating and renewing instances together; active
episodes are bounded separately by `--prover-sandbox-concurrency`. Creation
also respects `E2B_CREATE_CONCURRENCY`. SDK control requests use HTTP/1.1
so long image-creation calls have separate connections from renew/kill calls;
envd command traffic keeps its existing transport. Empty inventory waits for a bounded
time and becomes an infrastructure failure on timeout. Creation failures
retry with backoff. Idle sandboxes are renewed with a 900-second lifetime;
claims receive at least 3600 seconds, or the episode budget plus 300 seconds.

`RUN_ID` and an absolute `RUN_DIR` on shared storage are required. The
`.e2b-warm-pool.lock` file in that directory permits one owner across rollout
processes; a second owner fails before allocation. The launcher forwards
these settings into Ray. Checkpoint pauses retain and renew the clean pool.
Normal worker shutdown reclaims idle instances while resolving in-flight
allocations, releasing service capacity before waiting for pending creates.
Idle renewal and cleanup each use up to 64 concurrent requests. Independent
renewal workers let maintenance continue when one instance is slow.
Before allocation the worker checks its file-descriptor budget and raises the
soft limit when the existing hard limit permits it; an insufficient hard limit
fails immediately. The DLC head also reclaims this run/attempt's pooled instances
after stopping Ray, validating ownership metadata before each deletion. This
fallback cleanup has a 600-second default deadline, configurable through
`E2B_WARM_POOL_CLEANUP_TIMEOUT_SEC` on the DLC head.
An abrupt process/container kill or late server-side allocation still relies
on the provider TTL.

The existing sampling metrics include `sandbox_acquire_seconds_mean` and
`sandbox_pool_hit_ratio`. `sandbox_pool/*` reports ready/creating inventory,
creation and renewal failures, allocation/runtime-check/renewal creation-failure
counters, mean creation time, and recent claim p50/p95
over at most 1024 claims. First-fill time is logged separately and does not
consume the episode progress watchdog. Pool wait and creation still have
their own deadlines.

Measure refill rate before increasing training concurrency. Required idle
capacity depends on the claim burst and replenishment latency; a finite pool
cannot sustain consumption faster than the provider can create replacements.
The client pool's idle sandboxes have already been allocated by E2B and count
toward the service's allocated-instance quota.

## Gateway with the Aliyun provider

An E2B-compatible gateway selects its Alibaba Cloud backend using
`api_headers={"X-Sandbox-Provider": "aliyun"}`. For RL submission, configure:

```bash
--e2b-api-url https://api.gateway.example \
--e2b-domain gateway.example \
--e2b-sandbox-provider aliyun \
--e2b-key-file /absolute/path/to/gateway_sandbox_api_key \
--e2b-template YOUR_VERIFIED_ALIYUN_LEAN428_TEMPLATE_ID
```

Direct launcher/evaluation use sets `E2B_API_URL`, `E2B_DOMAIN`,
`E2B_SANDBOX_PROVIDER=aliyun`, `E2B_API_KEY_FILE`, and `E2B_TEMPLATE_ID`.
The key belongs to the Apodex gateway. Its underlying Alibaba Cloud account
and region are configured on the gateway; the client domain alone does not
identify them. Choose a template listed by the gateway with the Aliyun
provider header. Existing E2B-hosted template IDs are not transferable.

The launcher passes these non-secret routing settings into Ray; sandbox
creation sends the provider header through the SDK's `api_headers` option.
The SDK retains that API configuration for sandbox lifecycle calls, including
cleanup. Credentials remain in the staged 0600 file, outside Ray job JSON.

## Alibaba Cloud FC Agent Sandbox

FC Agent Sandbox uses the same E2B SDK, with its own API key and a template
built in the target region. The existing E2B-hosted Lean template ID cannot
be reused. For Singapore, pass these options to `rl.submit_dlc`:

```bash
--e2b-api-url https://api.ap-southeast-1.e2b.fc.aliyuncs.com \
--e2b-domain ap-southeast-1.e2b.fc.aliyuncs.com \
--e2b-key-file /absolute/path/to/aliyun_fc_sandbox_api_key \
--e2b-template YOUR_VERIFIED_FC_LEAN428_TEMPLATE_ID
```

For direct launcher or evaluation use, export `E2B_API_URL`, `E2B_DOMAIN`,
`E2B_API_KEY_FILE`, and `E2B_TEMPLATE_ID` with the corresponding values.
The launcher forwards the API URL and sandbox domain to Ray workers and
records both in `run-manifest.txt`, `preflight.json` and `e2b-preflight.json`.
Both endpoint variables must be set together; mismatched FC regions are
rejected. Choose a template built for that service. Credentials remain in the owner-owned
0600 file and are loaded inside each process.

Create the key in the [FC Agent Sandbox console](https://www.alibabacloud.com/help/en/functioncompute/create-api-key).
An ACR password or a RAM access key is a different credential. The pinned
E2B SDK 2.34.0 requires an `e2b_` prefix followed by hexadecimal characters.
The service must also accept the key; passing the local format check alone
is insufficient.

Build a template from a versioned `linux/amd64` Lean 4.28 image in an ACR
Enterprise Edition repository in the same region. Follow Alibaba's
[custom image template instructions](https://www.alibabacloud.com/help/en/functioncompute/build-a-custom-image-template)
for conversion and VPC requirements. The image must include the existing
`/opt/code-prover/prover` skeleton, Lean/Mathlib 4.28, `lean-lsp-mcp`, and
writable `/task`, `/tests` and `/logs` directories for the sandbox user.
Retain the current 2 vCPU / 8192 MiB baseline until real tests justify a change.
Before switching training, run the existing `python3 -m rl.e2b_smoke
--template TEMPLATE_ID`, then real positive and negative verifier trials.
E2B SDK compatibility does not by itself establish template compatibility.

## Render, then submit

Use an `aliyun` CLI profile authorized for your workspace and region.
The submitter sends the complete CreateJob JSON, including RDMA settings.

From this repository root. `CODEPROVER_PROJECT_ROOT` can override the default
project directory; otherwise it is inferred from an ancestor containing `miles/`
or falls back to the checkout's parent:

```bash
PROJECT_ROOT=/path/to/project
python3 -m rl.submit_dlc \
  --workspace-id "$DLC_WORKSPACE_ID" \
  --resource-id "$DLC_RESOURCE_ID" \
  --data-source-id "$DLC_DATA_SOURCE_ID" \
  --mount-path "$DLC_MOUNT_PATH" \
  --image "$CODEPROVER_RL_IMAGE" \
  --e2b-template "$E2B_TEMPLATE_ID" \
  --stage 1 \
  --name TRACES_Verification_RL_D3_stage01_YYYYMMDD \
  --run-id TRACES_Verification_RL_D3_YYYYMMDD \
  --codeprover-root "$PROJECT_ROOT/Code-Prover-V3" \
  --codeprover-commit "$(git rev-parse HEAD)" \
  --miles-root "$PROJECT_ROOT/miles" \
  --e2b-key-file "/secure/path/sandbox.key"
```

Without `--submit`, this writes the existing job body, parameter and path
inspection outputs only. If CUDA imports are unavailable on the control
machine, the output explicitly reports checkpoint validation as incomplete;
submission still requires it to pass. Runtime preflight also validates all
prompts and checkpoint compatibility, checks dependencies, and allocates and
destroys an E2B sandbox before model loading. The key is supplied through an
owner-owned, non-symlink `0600` file. Rollouts upload task/verifier files to E2B.

`run_prover_rl.sh` still supports the historical topologies, but
`submit_dlc.py` supports the configured 8x8 and 10x8 campaign topologies. The old GB300 wrappers
under Miles are not the submission entrypoint for this campaign.


### Fixed evaluation and sampling diagnostics

`PROVER_EVAL_PROMPT_DATA` enables fixed-set evaluation through the same episode
loop and verifier as training. The launcher defaults to `EVAL_INTERVAL=10`,
`N_SAMPLES_PER_EVAL_PROMPT=2`, and `PROVER_FULL_EVAL_SAMPLES_PER_PROMPT=8`.
Every 10 steps, all 256 validation tasks receive two trials (512 episodes),
reporting pass@1 and pass@2. At the end of each stage (`NUM_ROLLOUT`), all
256 tasks receive eight trials (2048 episodes), reporting pass@1, pass@2 and
pass@8. The full evaluation replaces the small evaluation at that step.
Full-evaluation pass@2 averages success over all pairs of the eight trials
per task, using `1 - (n-c)(n-c-1)/(n(n-1))`, where `c` is the success count.
Set `OBSERVE_TRAINING_ENTROPY=1` to enable entropy diagnostics.
When preparing a launch from an older submission body, replace its explicit
`N_SAMPLES_PER_EVAL_PROMPT=8` with `2` and use the updated CodeProver and Miles
sources. Existing running processes retain their launch configuration.
`PROVER_TUNING_PROMPT_DATA` reserves a separate set for budget comparisons.
The launcher rejects task-name or source-byte overlap with either reserved set.
Resume preserves optimizer and sampler state. Evaluation pauses the background
sampler, recycles in-flight work through its existing checkpoint path, and resumes
sampling afterwards; it does not apply the training dynamic filter.

Create a fixed hard-pool split with `python -m rl.make_eval_data --pool POOL.jsonl
--task-root DATA_ROOT --exclude-prompts batch-01.prompts.jsonl ...
--out validation.prompts.jsonl --tuning-out tuning.prompts.jsonl`.
The defaults reserve 256 validation tasks and 64 separate tuning tasks, balanced
between coding and math, excluding task names and identical task source from all
provided training batches. These are held out from this RL campaign, not evidence
of absence from the original model's pretraining. Keep both sets excluded from
future training batches.

`sampling/raw/*` describes healthy, non-stale completed groups before dynamic
filtering; `sampling/accepted/*` describes the groups retained for training.
Infrastructure failures and stale groups have separate counters. Group fractions
refer to attempts consumed by that collection pass, not unique dataset tasks.
`stop/<reason>/pass_rate` separates truncation from verifier failure. Full-context
token accounting includes the initial prompt, model-generated tokens, inserted
end-of-turn tokens, and tool/feedback messages including their formatting. Old
replayed samples without these fields are excluded from token fractions and
reported through `token_accounting_coverage`.

For an existing fixed-weight SGLang server, `python -m rl.evaluate` accepts
`--prompt-data`, `--router-host`, `--router-port`, `--out`, `--turn-budgets 2048 4096`,
`--samples-per-prompt 8`, and the same `--prover-*` settings as training. Use the
same model checkpoint, tasks, temperature and total-token budget for both arms.
Output records and summaries include every trial, including infrastructure
failures, without dynamic filtering. Do not change server weights during an eval.
With multiple budgets, trials are interleaved; measure each budget in a separate
run when comparing GPU-hour cost. `E2B_CREATE_CONCURRENCY` defaults to eight per
process/event loop; divide this creation limit across evaluation nodes to preserve
the campaign-wide allocation rate.


### Standalone checkpoint export

For this text-only Qwen RL campaign, first export the distributed checkpoint with
Miles `tools/convert_torch_dist_to_hf.py`, then run `python -m rl.repack_hf
--converted-dir EXPORTED_HF --original-dir SFT_HF --output-dir PACKED_HF`.
The second step restores the original packed-expert layout and copies only the
vision/MTP modules that were not part of RL training. Every trained language-model
parameter must come from the RL export; missing or unmapped weights fail the
conversion. The output index is written only after all shards pass validation.

## Recovering transient rollout failures

A failed or aborted sample makes its whole prompt group ineligible for training.
Groups are retried at most twice by default (`--prover-async-max-group-retries`,
launcher environment `PROVER_ASYNC_MAX_GROUP_RETRIES`). The retry count lives in
sample metadata and survives checkpoint/resume. After the retry budget is used,
the group is discarded and sampling continues. Successful samples are still
regenerated along with their group; individual-sample reuse is not enabled.
Normal zero-reward results are not infrastructure failures.

The consecutive-group fail-fast limit is disabled by default
(`PROVER_ASYNC_MAX_CONSECUTIVE_FAILURES=0`); a positive value explicitly restores
it. Fatal worker errors still propagate immediately. The existing no-progress
watchdog now also checks a nonempty queue of failed, filtered, or stale groups;
only an accepted group resets it. Both the launcher and direct Python entrypoint
default to the episode timeout plus 300 seconds: a 3300-second episode gives a
3600-second no-progress deadline. An explicit `PROVER_ASYNC_NO_PROGRESS_TIMEOUT_SEC`
or `--prover-async-no-progress-timeout-sec` overrides this default. This is a
no-progress limit, not a guarantee that every queued retry can finish before it;
long acquisition/queue delays may require a larger explicit value. Warm-pool
prefill has its own deadline.

E2B task/test uploads retry transient transport errors up to three total attempts
for each `mkdir -p` or identical file overwrite. Each safe operation has a
120-second deadline including reconnect/backoff. Each retry rebuilds the private
command/file transport while retaining the sandbox and files. Permission errors,
authentication errors and ordinary command exits are not retried. Arbitrary agent
commands and verifier commands are never automatically replayed; allocation and
bootstrap retain their existing bounded allocation retry policy.

Failure logs include sandbox ID, task/sample identity, phase, operation, turn,
tool and elapsed time, with a credential-redacted exception chain (including
implicit causes). Episode timeouts preserve the operation awaiting completion.
The verifier execution limit is 600 seconds; an upload's own request deadline
is reported separately from the outer episode timeout. These changes do not
change provider/ALB configuration or deployment snapshots.
