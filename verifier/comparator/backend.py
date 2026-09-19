"""Compare textual Lean exports from three independent, disposable containers.

Only candidate source is accepted from the agent. No agent lakefile, dependency,
olean, command output, or claimed target list is trusted by the final checker.
This backend requires a local Docker daemon and the dedicated comparator image.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import time
import uuid

from verifier.grade import check_spec_intact, find_forbidden_constructs, find_sorries

REVISION = 'd03acab154d269c06e60e4de7e4cc85deebff94b'
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_EXPORT_BYTES = 512 * 1024 * 1024


class ComparatorInfrastructureError(RuntimeError):
    """No score may be inferred: image, challenge, transport or runtime failed."""


async def _command(args, *, timeout=600, output=None, input_file=None, output_limit=1024 * 1024):
    """Bound diagnostic output, reap clients on cancellation, optionally stream an input file."""
    source = open(input_file, 'rb') if input_file else None
    dest = open(output, 'wb') if output else None
    async def drain(stream, limit, file=None):
        kept = bytearray()
        count = 0
        while chunk := await stream.read(65536):
            take = chunk[:max(0, limit - count)]
            count += len(take)
            if file is not None:
                file.write(take)
            else:
                kept.extend(take)
        return bytes(kept).decode(errors='replace')
    try:
        proc = await asyncio.create_subprocess_exec(*args, stdin=source,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout = asyncio.create_task(drain(proc.stdout, output_limit if dest else 65536, dest))
        stderr = asyncio.create_task(drain(proc.stderr, 65536))
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
            raise
        finally:
            out, err = await asyncio.gather(stdout, stderr)
        return proc.returncode, out if dest is None else '', err
    finally:
        if source:
            source.close()
        if dest:
            dest.close()


async def _checked(args, **kwargs):
    code, out, err = await _command(args, **kwargs)
    if code:
        raise ComparatorInfrastructureError(f'{args[:2]} exited {code}: {err[-2000:]}')
    return out


async def _container(image, phase, inputs, outputs, *, timeout):
    name = 'codeprover-judge-' + uuid.uuid4().hex
    # No writable host mounts, sockets, network, privileges or shared process namespace.
    # /work and /input are private tmpfs. No host paths are mounted. Each
    # phase receives only its own inputs; comparison never executes Lean code.
    args = ['docker', 'run', '-d', '--name', name, '--network=none', '--read-only',
            '--user=65532:65532',
            '--pids-limit=256', '--memory=8g', '--cpus=4', '--ulimit', 'fsize=536870912:536870912',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m,mode=1777',
            '--tmpfs', '/work:rw,nosuid,nodev,size=2g,mode=1777',
            '--tmpfs', '/input:rw,nosuid,nodev,size=1100m,mode=1777',
            '--entrypoint', '/bin/sleep', image, 'infinity']
    try:
        await _checked(args, timeout=120)
        for path in inputs.iterdir():
            await _checked(['docker', 'exec', '-i', name, 'python3', '-c',
                            'import shutil,sys; f=open(sys.argv[1], "wb"); shutil.copyfileobj(sys.stdin.buffer,f)',
                            f'/input/{path.name}'], input_file=path, timeout=120)
        log = outputs / (phase + '.log')
        code, _, err = await _command(
            ['docker', 'exec', name, 'python3', '/opt/codeprover/phase.py', phase],
            timeout=timeout, output=log)
        (outputs / (phase + '.stderr')).write_text(err)
        if code == 0 and phase != 'compare':
            for source, target in [('export.ndjson', phase + '.ndjson')] + (
                    [('targets.json', 'targets.json')] if phase == 'challenge' else []):
                reader = (
                    'import os,stat,sys; '
                    'fd=os.open(sys.argv[1], os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK); '
                    's=os.fstat(fd); '
                    f'assert stat.S_ISREG(s.st_mode) and s.st_size <= {MAX_EXPORT_BYTES}; '
                    f'f=os.fdopen(fd, "rb"); sys.stdout.buffer.write(f.read({MAX_EXPORT_BYTES + 1}))'
                )
                await _checked(['docker', 'exec', name, 'python3', '-c', reader, f'/work/{source}'],
                               output=outputs / target, output_limit=MAX_EXPORT_BYTES + 1, timeout=120)
                path = outputs / target
                # Treat symlinks and huge/corrupt files as failure before consuming them.
                if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EXPORT_BYTES:
                    raise ComparatorInfrastructureError('invalid exported artifact')
        return code
    finally:
        # Shield cleanup when the episode is cancelled. No successful judge container is reused.
        cleanup = asyncio.create_task(_checked(['docker', 'rm', '-f', name], timeout=60))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise


def source_checks(original: str, final: str) -> dict:
    trusted_spec = check_spec_intact(original, original, strict_bytes=True)
    if not trusted_spec.get('ok'):
        raise ComparatorInfrastructureError('malformed trusted benchmark markers')
    spec = check_spec_intact(original, final, strict_bytes=True)
    if not spec.get('ok'):
        # EOF line terminators cannot change a successfully compiled Lean term.
        # Compare only this narrow normal form; retain every other protected byte,
        # including whitespace inside strings and all internal blank lines.
        eof_spec = check_spec_intact(original.rstrip('\r\n'), final.rstrip('\r\n'), strict_bytes=True)
        if eof_spec.get('ok'):
            spec = {**eof_spec, 'mode': 'byte_exact_except_eof_newlines'}
    # Definition holes need the surrounding immutable contract; markerless tasks
    # must be normalized with trusted markers before enabling this backend.
    if not spec.get('checked'):
        raise ComparatorInfrastructureError('comparator requires trusted benchmark markers')
    return {'spec_intact': bool(spec['ok']), 'sorry_free': not find_sorries(final),
            'forbidden_free': not find_forbidden_constructs(final), 'spec_detail': spec}


async def judge(original: str, final: str, *, image: str, artifacts: Path,
                timeout: int = 1200, challenge: str | None = None) -> dict:
    """Persist source/results, pin a local image ID, and return an independent verdict.

    Invalid proofs are rejected. Infrastructure failures raise (also recorded);
    callers must not turn an unavailable comparator into an ordinary zero score.
    """
    if not image or timeout <= 0:
        raise ValueError('comparator image and positive timeout required')
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (artifacts / 'original.lean').write_bytes(original.encode())
    (artifacts / 'solution.lean').write_bytes(final.encode())
    details = {'comparator_revision': REVISION, 'lean_version': '4.28.0',
               'source_sha256': hashlib.sha256(final.encode()).hexdigest(),
               'artifact_dir': str(artifacts), 'status': 'infrastructure_error',
               'resource_limits': {'memory_bytes': 8 * 1024**3, 'cpus': 4, 'pids': 256}}
    try:
        if max(len(original.encode()), len(final.encode())) > MAX_SOURCE_BYTES:
            raise ComparatorInfrastructureError('source exceeds 2 MiB transport limit')
        trusted = original if challenge is None else challenge
        if not check_spec_intact(original, trusted, strict_bytes=True).get('ok'):
            raise ComparatorInfrastructureError('trusted challenge changed protected task bytes')
        if len(trusted.encode()) > MAX_SOURCE_BYTES:
            raise ComparatorInfrastructureError('trusted challenge exceeds source size limit')
        details['challenge_sha256'] = hashlib.sha256(trusted.encode()).hexdigest()
        details['challenge_mode'] = 'original' if challenge is None else 'trusted_override'
        (artifacts / 'challenge.lean').write_bytes(trusted.encode())
        checks = source_checks(original, final)
        details['source_checks'] = checks
        if not all(checks[k] for k in ('spec_intact', 'sorry_free', 'forbidden_free')):
            details.update(status='rejected', accepted=False, reason='source_contract')
            return details
        image_id = (await _checked(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], timeout=30)).strip()
        if not image_id.startswith('sha256:'):
            raise ComparatorInfrastructureError('invalid comparator image ID')
        details['image_id'] = image_id
        # Separate phase inputs: solution code cannot see or mutate challenge output.
        with tempfile.TemporaryDirectory(prefix='codeprover-judge-') as temp:
            root = Path(temp)
            root.chmod(0o755)
            for name in ('challenge', 'solution', 'compare'):
                (root / name).mkdir(mode=0o755)
            out = artifacts
            (root / 'challenge/Main.lean').write_bytes(trusted.encode())
            (root / 'solution/Main.lean').write_bytes(final.encode())
            async with asyncio.timeout(timeout):
                code = await _container(image_id, 'challenge', root / 'challenge', out, timeout=timeout)
                if code:
                    raise ComparatorInfrastructureError(f'trusted challenge failed ({code})')
                config = json.loads((out / 'targets.json').read_text())
                if not config.get('theorem_names'):
                    raise ComparatorInfrastructureError('empty trusted theorem targets')
                details['targets'] = config
                (artifacts / 'targets.json').write_text(json.dumps(config, indent=2))
                (root / 'solution/targets.json').write_text(json.dumps(config))
                code = await _container(image_id, 'solution', root / 'solution', out, timeout=timeout)
                if code == 12:
                    details.update(status='rejected', accepted=False, reason='solution_memory_limit')
                elif code in (10, 11):
                    details.update(status='rejected', accepted=False, reason='solution_build_or_export')
                elif code:
                    raise ComparatorInfrastructureError(f'solution runtime failed ({code})')
                else:
                    for source, target in [('challenge.ndjson', 'challenge.ndjson'),
                                           ('solution.ndjson', 'solution.ndjson'), ('targets.json', 'targets.json')]:
                        if source == 'targets.json':
                            (root / 'compare' / target).write_bytes((out / source).read_bytes())
                        else:
                            shutil.move(str(out / source), str(root / 'compare' / target))
                    code = await _container(image_id, 'compare', root / 'compare', out, timeout=timeout)
                    if code not in (0, 1):
                        raise ComparatorInfrastructureError(f'comparator runtime failed ({code})')
                    details.update(status='accepted' if code == 0 else 'rejected',
                                   accepted=code == 0, reason='comparator')
            # Raw exports are large; source, target list and logs are durable.
            for exported in artifacts.glob('*.ndjson'):
                exported.unlink()
        return details
    except BaseException as exc:
        details['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        details['duration_sec'] = time.monotonic() - started
        (artifacts / 'comparator.json').write_text(json.dumps(details, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--solution', type=Path, required=True)
    p.add_argument('--challenge', type=Path, help='optional task-author baseline; never candidate-supplied')
    p.add_argument('--image', required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    args = p.parse_args()
    result = asyncio.run(judge(args.original.read_bytes().decode(), args.solution.read_bytes().decode(),
                              image=args.image, artifacts=args.artifacts, timeout=args.timeout,
                              challenge=args.challenge.read_bytes().decode() if args.challenge else None))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['accepted'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
