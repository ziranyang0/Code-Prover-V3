"""Durable shared-filesystem transport from DLC workers to a CPU/Docker judge.

Only trusted rollout processes may write this directory. Agent sandboxes must
never mount it. The CPU service resolves originals from a pinned task catalog;
requests contain candidate bytes and hashes, never commands or executable paths.
"""
from __future__ import annotations

import argparse
import asyncio
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import uuid

from verifier.comparator.backend import ComparatorInfrastructureError, MAX_SOURCE_BYTES, judge
from verifier.grade import check_spec_intact

REQUEST_ID = re.compile(r'^[0-9a-f]{32}$')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_bytes(path: Path, limit=MAX_SOURCE_BYTES):
    # Cross-node CPFS readers can get ESTALE while service.json is atomically
    # replaced. Reopen the path a bounded number of times; retain all validation
    # and let unrelated I/O errors (including ENOSPC) surface immediately.
    # CPFS can return EIO on reopen directly after ESTALE; only that observed
    # recovery sequence is retried, with six attempts / 1.55 seconds of backoff.
    saw_stale = False
    for attempt in range(6):
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
                raise ComparatorInfrastructureError(f'invalid queue artifact: {path.name}')
            with path.open('rb') as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ComparatorInfrastructureError('queue artifact exceeds size limit')
            return data
        except OSError as exc:
            # A buffered reader can raise EIO while closing after its read raised
            # ESTALE. Preserve that original failure even when __exit__ masks it.
            context = exc
            seen = set()
            while context is not None and id(context) not in seen:
                seen.add(id(context))
                saw_stale = saw_stale or (isinstance(context, OSError) and context.errno == errno.ESTALE)
                context = context.__cause__ or context.__context__
            retryable = exc.errno == errno.ESTALE or (saw_stale and exc.errno == errno.EIO)
            if not retryable or attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def healthy(queue: Path, *, max_age=30):
    try:
        status = json.loads(read_bytes(queue / 'service.json', 65536))
        if status['state'] != 'ready' or time.time() - status['heartbeat'] > max_age:
            raise ValueError('worker not ready or heartbeat stale')
        return status
    except (OSError, KeyError, ValueError) as exc:
        raise ComparatorInfrastructureError(
            f'CPU comparator service unavailable: {type(exc).__name__}'
            + (f' (errno={exc.errno})' if isinstance(exc, OSError) else f' ({exc})')) from exc


async def submit(original: str, final: str, *, queue: Path, timeout=3600, poll_sec=1.0):
    queue = queue.resolve()
    status = healthy(queue)
    original_bytes, final_bytes = original.encode(), final.encode()
    if max(len(original_bytes), len(final_bytes)) > MAX_SOURCE_BYTES:
        raise ComparatorInfrastructureError('source exceeds queue size limit')
    request_id = uuid.uuid4().hex
    directory = queue / 'requests' / request_id
    directory.mkdir(mode=0o700)
    (directory / 'solution.lean').write_bytes(final_bytes)
    request = {'id': request_id, 'original_sha256': digest(original_bytes),
               'source_sha256': digest(final_bytes), 'created_at': time.time(),
               'deadline': time.time() + timeout, 'catalog_sha256': status['catalog_sha256']}
    atomic_json(directory / 'request.json', request)  # publication is the last write
    deadline = time.monotonic() + timeout
    try:
        while True:
            if (directory / 'result.json').exists():
                result = json.loads(read_bytes(directory / 'result.json', 2 * 1024 * 1024))
                if (result.get('id'), result.get('source_sha256'), result.get('original_sha256')) != (
                        request_id, request['source_sha256'], request['original_sha256']):
                    raise ComparatorInfrastructureError('CPU comparator result identity mismatch')
                if result.get('status') == 'infrastructure_error':
                    raise ComparatorInfrastructureError(result.get('error', 'CPU verifier failed'))
                verdict = result['verdict']
                if type(verdict.get('accepted')) is not bool:
                    raise ComparatorInfrastructureError('CPU verifier published invalid verdict')
                return {**verdict, 'queue_request_id': request_id, 'queue_dir': str(queue)}
            if time.monotonic() >= deadline:
                raise ComparatorInfrastructureError('CPU comparator queue deadline exceeded; source retained')
            healthy(queue)
            await asyncio.sleep(poll_sec)
    except BaseException:
        atomic_json(directory / 'cancel.json', {'time': time.time()})
        raise


def make_catalog(task_root: Path, destination: Path):
    entries = {}
    for path in sorted(task_root.glob('*/*/tests/original.lean')) + sorted(task_root.glob('*/tests/original.lean')):
        data = read_bytes(path)
        key = digest(data)
        entries.setdefault(key, {'source': data.decode('utf-8'), 'tasks': []})['tasks'].append(str(path.parent.parent))
    if not entries:
        raise ValueError('no trusted tasks found')
    atomic_json(destination, entries)
    return len(entries)


class Service:
    def __init__(self, queue: Path, catalog: Path, image: str, concurrency: int, timeout: int):
        self.queue = queue.resolve()
        self.catalog_bytes = read_bytes(catalog, 32 * 1024 * 1024)
        self.catalog_sha = digest(self.catalog_bytes)
        self.catalog = json.loads(self.catalog_bytes)
        for key, entry in self.catalog.items():
            if key != digest(entry['source'].encode()):
                raise ValueError('catalog source hash mismatch')
            if 'challenge_source' in entry and not check_spec_intact(
                    entry['source'], entry['challenge_source'], strict_bytes=True).get('ok'):
                raise ValueError('catalog challenge changed protected task bytes')
        self.image = image
        self.concurrency = concurrency
        self.timeout = timeout
        self.stopping = asyncio.Event()
        self.active = {}
        self.completed = set()
        self.started = time.time()

    async def process(self, directory: Path):
        request = None
        try:
            request = json.loads(read_bytes(directory / 'request.json', 65536))
            if not isinstance(request, dict):
                request = None
                raise ValueError('request must be an object')
            if request.get('id') != directory.name or not REQUEST_ID.fullmatch(directory.name):
                raise ValueError('invalid request ID')
            if request['catalog_sha256'] != self.catalog_sha:
                raise ValueError('task catalog changed')
            entry = self.catalog[request['original_sha256']]
            original = entry['source']
            challenge_args = {'challenge': entry['challenge_source']} if 'challenge_source' in entry else {}
            source = read_bytes(directory / 'solution.lean')
            if digest(source) != request['source_sha256']:
                raise ValueError('candidate source hash mismatch')
            if time.time() > request['deadline'] or (directory / 'cancel.json').exists():
                raise ValueError('request expired or cancelled')
            atomic_json(directory / 'running.json', {'pid': os.getpid(), 'started': time.time()})
            verdict = None
            # Infrastructure retries reuse these exact bytes; never resample the model.
            for attempt in range(2):
                try:
                    verdict = await judge(original, source.decode('utf-8'), image=self.image,
                        artifacts=directory / ('attempt-' + uuid.uuid4().hex), timeout=self.timeout, **challenge_args)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if attempt == 1 or time.time() > request['deadline']:
                        raise
                    await asyncio.sleep(2)
            result = {'status': 'completed', 'verdict': verdict}
        except asyncio.CancelledError:
            # Restarting a stopped service can reclaim the request. judge() cleans containers.
            raise
        except Exception as exc:
            result = {'status': 'infrastructure_error', 'error': f'{type(exc).__name__}: {exc}'}
        if request is not None:
            result.update({key: request.get(key) for key in ('id', 'original_sha256', 'source_sha256')})
        else:
            result['id'] = directory.name
        result['completed_at'] = time.time()
        atomic_json(directory / 'result.json', result)

    def pending(self, active, capacity):
        """Scan shared storage off the event loop, skipping known finished requests."""
        pending = []
        for directory in sorted((self.queue / 'requests').iterdir()):
            if len(pending) >= capacity:
                break
            if (not REQUEST_ID.fullmatch(directory.name) or directory.name in active
                    or directory.name in self.completed):
                continue
            if not directory.is_dir() or directory.is_symlink():
                continue
            if (directory / 'result.json').exists():
                self.completed.add(directory.name)
            elif (directory / 'request.json').exists():
                pending.append(directory)
        return pending

    async def heartbeat(self):
        try:
            while not self.stopping.is_set():
                await asyncio.to_thread(atomic_json, self.queue / 'service.json', {
                    'state': 'ready', 'pid': os.getpid(), 'heartbeat': time.time(), 'started': self.started,
                    'active': len(self.active), 'concurrency': self.concurrency,
                    'catalog_sha256': self.catalog_sha, 'image_id': self.image})
                try:
                    await asyncio.wait_for(self.stopping.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
        except BaseException:
            self.stopping.set()
            raise

    async def run(self):
        self.queue.mkdir(parents=True, exist_ok=True)
        (self.queue / 'requests').mkdir(exist_ok=True)
        # Kernel-owned advisory lock is released on crashes; only one dispatcher per queue.
        with (self.queue / 'service.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            from verifier.comparator.backend import _checked
            self.image = (await _checked(['docker', 'image', 'inspect', '--format', '{{.Id}}', self.image], timeout=30)).strip()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.stopping.set)
            heartbeat = asyncio.create_task(self.heartbeat())
            try:
                while not self.stopping.is_set():
                    for name, task in list(self.active.items()):
                        if (self.queue / 'requests' / name / 'cancel.json').exists() and not task.done():
                            task.cancel()
                        if task.done():
                            if not task.cancelled():
                                task.result()
                            del self.active[name]
                    capacity = self.concurrency - len(self.active)
                    if capacity:
                        pending = await asyncio.to_thread(self.pending, set(self.active), capacity)
                        if not self.stopping.is_set():
                            for directory in pending:
                                self.active[directory.name] = asyncio.create_task(self.process(directory))
                    try:
                        await asyncio.wait_for(self.stopping.wait(), timeout=1)
                    except asyncio.TimeoutError:
                        pass
            finally:
                self.stopping.set()
                try:
                    # Let any in-flight heartbeat write finish before publishing stopped.
                    await heartbeat
                finally:
                    for task in self.active.values():
                        task.cancel()
                    await asyncio.gather(*self.active.values(), return_exceptions=True)
                    atomic_json(self.queue / 'service.json', {'state': 'stopped', 'pid': os.getpid(), 'heartbeat': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    catalog = commands.add_parser('catalog')
    catalog.add_argument('--task-root', type=Path, required=True)
    catalog.add_argument('--output', type=Path, required=True)
    serve = commands.add_parser('serve')
    serve.add_argument('--queue', type=Path, required=True)
    serve.add_argument('--catalog', type=Path, required=True)
    serve.add_argument('--image', required=True)
    serve.add_argument('--concurrency', type=int, default=4)
    serve.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    if args.command == 'catalog':
        print(json.dumps({'unique_originals': make_catalog(args.task_root, args.output)}))
    else:
        if args.concurrency < 1 or args.timeout < 1:
            parser.error('positive concurrency and timeout required')
        asyncio.run(Service(args.queue, args.catalog, args.image, args.concurrency, args.timeout).run())

if __name__ == '__main__':
    main()
