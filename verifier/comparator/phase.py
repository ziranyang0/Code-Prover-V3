"""Trusted image entrypoint. Candidate compilation runs only in a disposable container."""
import ctypes
import os
import json
import pathlib
import shutil
import subprocess
import sys
import time
import resource

BIN = pathlib.Path('/opt/codeprover/bin')
work = pathlib.Path('/work')

def memory_events():
    """Read the container's kernel-owned counters; unavailable counters stay unknown."""
    names = ('/sys/fs/cgroup/memory.events', '/sys/fs/cgroup/memory.max',
             '/sys/fs/cgroup/memory/memory.oom_control',
             '/sys/fs/cgroup/memory/memory.limit_in_bytes',
             '/sys/fs/cgroup/memory/memory.failcnt',
             '/sys/fs/cgroup/memory/memory.max_usage_in_bytes')
    values = {}
    for name in names:
        try:
            values[name] = pathlib.Path(name).read_text().strip()
        except OSError:
            pass
    return values

def oom_kills(events):
    for path in ('/sys/fs/cgroup/memory.events',
                 '/sys/fs/cgroup/memory/memory.oom_control'):
        value = events.get(path)
        if value is not None:
            try:
                return int(dict(line.split() for line in value.splitlines())['oom_kill'])
            except (ValueError, KeyError):
                pass
    return None


def run(args):
    started = time.monotonic()
    code = subprocess.run(args, cwd=work).returncode
    print(json.dumps({'command': args[0], 'exit_code': code,
                      'duration_sec': time.monotonic() - started,
                      'max_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                      'memory_events': memory_events()}), flush=True)
    return code

def main():
    # DSW rejects the entire Docker --security-opt flag family. Set the same
    # restrictive no_new_privs bit before any Lean/native candidate code runs;
    # it is inherited by all children and cannot be cleared. DSW also disallows
    # --cap-drop. The non-root process has no effective/permitted/inheritable/
    # ambient capabilities; no_new_privs prevents gaining any on exec.
    if os.geteuid() == 0:
        raise RuntimeError('judge phases must not run as root')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')
    status = dict(line.split(':', 1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if any(int(status[key].strip(), 16) for key in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb')) or status['NoNewPrivs'].strip() != '1':
        raise RuntimeError('judge requires zero capabilities and no_new_privs')
    phase = sys.argv[1]
    if phase == 'compare':
        return run([str(BIN / 'codeprover_comparator'), '/input/targets.json',
                    '/input/challenge.ndjson', '/input/solution.ndjson'])
    if phase not in ('challenge', 'solution'):
        raise ValueError('unknown phase')
    if pathlib.Path('/task/lean-toolchain').read_text().strip() != 'leanprover/lean4:v4.28.0':
        raise RuntimeError('image must use Lean/Mathlib 4.28')
    # Use only the baked, read-only oleans. Calling `lake build` would let Lake
    # try to reconcile dependency git URLs and mutate the shared package tree.
    packages = sorted(pathlib.Path('/task/.lake/packages').glob('*/.lake/build/lib/lean'))
    if not any((p / 'Mathlib.olean').is_file() for p in packages):
        raise RuntimeError('trusted Mathlib cache missing')
    os.environ['LEAN_PATH'] = ':'.join([str(work), '/opt/codeprover/lean', *(str(p) for p in packages)])
    # Scope a compiler cache around protected declarations. This prevents
    # editable code from changing which identical matcher the fixed spec reuses.
    # Source contract checks and hashes always use the unmodified captured bytes.
    raw = pathlib.Path('/input/Main.lean').read_bytes().decode('utf-8')
    lines = ['import CodeProverCompile\n']
    boundaries = ('precond_aux', 'postcond_aux')
    for line in raw.splitlines(keepends=True):
        if any(line.strip() == '-- !benchmark @start ' + name for name in boundaries):
            lines.append('run_cmd CodeProver.beginSpecification\n')
        if any(line.strip() == '-- !benchmark @start ' + name for name in ('code_aux', 'proof_aux')):
            lines.append('run_cmd CodeProver.endSpecification\n')
        lines.append(line)
    (work / 'Main.lean').write_bytes(''.join(lines).encode('utf-8'))
    before_oom = oom_kills(memory_events())
    compile_code = run(['lean', '-o', '/work/Main.olean', '/work/Main.lean'])
    after_oom = oom_kills(memory_events())
    if compile_code == -9 and before_oom is not None and after_oom is not None and after_oom > before_oom:
        return 12  # this candidate exhausted its isolated container's memory budget
    if compile_code:
        return 10 if compile_code == 1 else 20  # candidate compile rejection; challenge maps this to infra
    if phase == 'challenge':
        with (work / 'targets.json').open('w') as out:
            subprocess.run([str(BIN / 'codeprover_targets')], cwd=work,
                           stdout=out, check=True)
    else:
        shutil.copyfile('/input/targets.json', work / 'targets.json')
    config = json.loads((work / 'targets.json').read_text())
    primitives = json.loads(subprocess.check_output([str(BIN / 'codeprover_comparator'), '--primitives']))
    names = list(dict.fromkeys(config['theorem_names'] + config['definition_names'] + primitives))
    with (work / 'export.ndjson').open('w') as out:
        proc = subprocess.run([str(BIN / 'lean4export'), 'Main', '--', *names],
                              cwd=work, stdout=out)
    return 0 if proc.returncode == 0 else (11 if proc.returncode == 1 else 20)

if __name__ == '__main__':
    try:
        result = main()
    except Exception:
        import traceback
        traceback.print_exc()
        result = 20  # image/setup failures must never look like a proof rejection
    sys.exit(result)
