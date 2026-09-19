import asyncio
import errno
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

from verifier.comparator.queue import Service, atomic_json, digest, healthy, submit
from verifier.comparator.backend import ComparatorInfrastructureError

ORIGINAL = 'theorem t : True := by\n  -- !benchmark @start proof\n  sorry\n  -- !benchmark @end proof\n'
SOLUTION = ORIGINAL.replace('sorry', 'trivial')

class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.queue = self.root / 'queue'
        (self.queue / 'requests').mkdir(parents=True)
        self.catalog = self.root / 'catalog.json'
        atomic_json(self.catalog, {digest(ORIGINAL.encode()): {'source': ORIGINAL, 'tasks': ['trusted']}})
        self.service = Service(self.queue, self.catalog, 'image', 2, 60)
        atomic_json(self.queue / 'service.json', {'state':'ready','heartbeat':time.time(),
                    'catalog_sha256':self.service.catalog_sha})

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def request(self):
        client = asyncio.create_task(submit(ORIGINAL, SOLUTION, queue=self.queue, timeout=5, poll_sec=.01))
        await asyncio.sleep(.02)
        directory = next((self.queue / 'requests').iterdir())
        return client, directory

    async def test_roundtrip_preserves_source_and_identity(self):
        client, directory = await self.request()
        judge = AsyncMock(return_value={'accepted':True,'status':'accepted'})
        with patch('verifier.comparator.queue.judge', judge):
            await self.service.process(directory)
        result = await client
        self.assertTrue(result['accepted'])
        self.assertEqual(judge.call_args.args, (ORIGINAL, SOLUTION))
        self.assertEqual(result['queue_request_id'], directory.name)

    async def test_challenge_override_only_comes_from_trusted_catalog(self):
        self.service.catalog[digest(ORIGINAL.encode())]['challenge_source'] = SOLUTION
        client, directory = await self.request()
        request = json.loads((directory/'request.json').read_text())
        request['challenge_source'] = ORIGINAL.replace('True', 'False')
        atomic_json(directory/'request.json', request)
        judge = AsyncMock(return_value={'accepted': True, 'status': 'accepted'})
        with patch('verifier.comparator.queue.judge', judge):
            await self.service.process(directory)
        self.assertTrue((await client)['accepted'])
        self.assertEqual(judge.call_args.kwargs['challenge'], SOLUTION)

    async def test_catalog_override_cannot_change_protected_statement(self):
        atomic_json(self.catalog, {digest(ORIGINAL.encode()): {
            'source': ORIGINAL, 'tasks': ['trusted'],
            'challenge_source': ORIGINAL.replace('True', 'False'),
        }})
        with self.assertRaisesRegex(ValueError, 'protected task bytes'):
            Service(self.queue, self.catalog, 'image', 2, 60)

    async def test_unknown_original_cannot_be_supplied_by_candidate(self):
        client, directory = await self.request()
        request=json.loads((directory/'request.json').read_text())
        request['original_sha256']='0'*64
        atomic_json(directory/'request.json',request)
        judge=AsyncMock()
        with patch('verifier.comparator.queue.judge',judge):
            await self.service.process(directory)
        with self.assertRaises(ComparatorInfrastructureError): await client
        judge.assert_not_awaited()

    async def test_changed_source_rejected_before_execution(self):
        client, directory = await self.request()
        (directory/'solution.lean').write_text('tampered')
        judge=AsyncMock()
        with patch('verifier.comparator.queue.judge',judge):
            await self.service.process(directory)
        with self.assertRaises(ComparatorInfrastructureError): await client
        judge.assert_not_awaited()

    async def test_backend_failure_retries_same_proof_not_generation(self):
        client, directory = await self.request()
        judge=AsyncMock(side_effect=[RuntimeError('temporary'), {'accepted':False,'status':'rejected'}])
        with patch('verifier.comparator.queue.judge',judge):
            await self.service.process(directory)
        self.assertFalse((await client)['accepted'])
        self.assertEqual(judge.await_count, 2)
        self.assertEqual(judge.call_args_list[0].args,judge.call_args_list[1].args)

    async def test_cancel_is_persisted_and_prevents_execution(self):
        client, directory = await self.request()
        client.cancel()
        with self.assertRaises(asyncio.CancelledError): await client
        self.assertTrue((directory/'cancel.json').exists())
        judge=AsyncMock()
        with patch('verifier.comparator.queue.judge',judge):
            await self.service.process(directory)
        judge.assert_not_awaited()

    async def test_stale_service_fails_before_publication(self):
        atomic_json(self.queue/'service.json', {'state':'ready','heartbeat':time.time()-100})
        with self.assertRaises(ComparatorInfrastructureError):
            await submit(ORIGINAL,SOLUTION,queue=self.queue,timeout=1)
        self.assertEqual(list((self.queue/'requests').iterdir()),[])

    async def test_estale_read_reopens_and_preserves_freshness_checks(self):
        path = self.queue / 'service.json'
        with path.open('rb') as stream, patch.object(Path, 'open', side_effect=[
                OSError(errno.ESTALE, 'Stale file handle'), stream]) as opened, \
                patch('verifier.comparator.queue.time.sleep') as sleep:
            self.assertEqual(healthy(self.queue)['catalog_sha256'], self.service.catalog_sha)
        self.assertEqual(opened.call_count, 2)
        sleep.assert_called_once_with(0.05)
        atomic_json(path, {'state': 'ready', 'heartbeat': time.time() - 100})
        with path.open('rb') as stream, patch.object(Path, 'open', side_effect=[
                OSError(errno.ESTALE, 'Stale file handle'), stream]), \
                patch('verifier.comparator.queue.time.sleep'):
            with self.assertRaises(ComparatorInfrastructureError):
                healthy(self.queue)

    async def test_cpfs_estale_then_eio_recovers_without_relaxing_freshness(self):
        path = self.queue / 'service.json'
        with path.open('rb') as stream, patch.object(Path, 'open', side_effect=[
                OSError(errno.ESTALE, 'Stale file handle'), OSError(errno.EIO, 'Input/output error'), stream]) as opened, \
                patch('verifier.comparator.queue.time.sleep') as sleep:
            self.assertEqual(healthy(self.queue)['catalog_sha256'], self.service.catalog_sha)
        self.assertEqual(opened.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        with patch.object(Path, 'open', side_effect=OSError(errno.EIO, 'Input/output error')) as opened, \
                patch('verifier.comparator.queue.time.sleep') as sleep:
            with self.assertRaises(ComparatorInfrastructureError):healthy(self.queue)
        opened.assert_called_once()
        sleep.assert_not_called()

    async def test_eio_during_close_does_not_mask_retryable_stale_read(self):
        class BrokenStream:
            def __enter__(self): return self
            def read(self, size): raise OSError(errno.ESTALE, 'Stale file handle')
            def __exit__(self, kind, value, trace): raise OSError(errno.EIO, 'Input/output error')
        path = self.queue / 'service.json'
        with path.open('rb') as stream, patch.object(Path, 'open', side_effect=[BrokenStream(), stream]) as opened, \
                patch('verifier.comparator.queue.time.sleep') as sleep:
            self.assertEqual(healthy(self.queue)['catalog_sha256'], self.service.catalog_sha)
        self.assertEqual(opened.call_count, 2)
        sleep.assert_called_once_with(0.05)

    async def test_persistent_estale_is_bounded_and_remains_infrastructure_error(self):
        with patch.object(Path, 'open', side_effect=OSError(errno.ESTALE, 'Stale file handle')) as opened, \
                patch('verifier.comparator.queue.time.sleep') as sleep:
            with self.assertRaises(ComparatorInfrastructureError) as caught:
                healthy(self.queue)
        self.assertEqual(opened.call_count, 6)
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual(caught.exception.__cause__.errno, errno.ESTALE)
        self.assertEqual(list((self.queue / 'requests').iterdir()), [])

    async def test_other_io_errors_are_not_retried(self):
        with patch.object(Path, 'open', side_effect=OSError(errno.EACCES, 'Permission denied')) as opened, \
                patch('verifier.comparator.queue.time.sleep') as sleep:
            with self.assertRaises(ComparatorInfrastructureError):
                healthy(self.queue)
        opened.assert_called_once()
        sleep.assert_not_called()

    async def test_forged_result_identity_rejected(self):
        client,directory=await self.request()
        atomic_json(directory/'result.json', {'id':directory.name,'source_sha256':'wrong',
                    'status':'completed','verdict':{'accepted':True}})
        with self.assertRaises(ComparatorInfrastructureError): await client

    async def test_slow_directory_scan_does_not_stall_heartbeat(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_scan(active, capacity):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test scan was not released')
            return []
        with patch.object(self.service, 'pending', blocked_scan), \
                patch('verifier.comparator.backend._checked', AsyncMock(return_value='image')):
            runner = asyncio.create_task(self.service.run())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                initial = healthy(self.queue)['heartbeat']
                deadline = time.monotonic() + 3
                while healthy(self.queue)['heartbeat'] <= initial:
                    self.assertLess(time.monotonic(), deadline)
                    await asyncio.sleep(.05)
                self.assertFalse(release.is_set())
                self.assertFalse(runner.done())
            finally:
                self.service.stopping.set()
                release.set()
                await asyncio.wait_for(runner, 3)
        self.assertEqual(json.loads((self.queue/'service.json').read_text())['state'], 'stopped')

    async def test_finished_request_scan_avoids_repeated_filesystem_probes(self):
        completed = self.queue/'requests'/('a'*32)
        completed.mkdir()
        atomic_json(completed/'request.json', {})
        atomic_json(completed/'result.json', {})
        self.assertEqual(self.service.pending(set(), 2), [])
        with patch.object(Path, 'is_dir', side_effect=AssertionError('revisited completed request')):
            self.assertEqual(self.service.pending(set(), 2), [])
        pending = self.queue/'requests'/('b'*32)
        pending.mkdir()
        atomic_json(pending/'request.json', {})
        self.assertEqual(self.service.pending(set(), 1), [pending])
        self.assertEqual(self.service.pending({pending.name}, 1), [])

    async def test_scan_failure_stops_service_and_heartbeat(self):
        with patch.object(self.service, 'pending', side_effect=OSError(errno.EIO, 'scan failed')), \
                patch('verifier.comparator.backend._checked', AsyncMock(return_value='image')):
            with self.assertRaisesRegex(OSError, 'scan failed'):
                await self.service.run()
        self.assertTrue(self.service.stopping.is_set())
        self.assertEqual(json.loads((self.queue/'service.json').read_text())['state'], 'stopped')

    async def test_heartbeat_write_failure_stops_dispatcher(self):
        def write_status(path, value):
            if value.get('state') == 'ready':
                raise OSError(errno.EIO, 'heartbeat write failed')
            return atomic_json(path, value)
        with patch('verifier.comparator.queue.atomic_json', side_effect=write_status), \
                patch('verifier.comparator.backend._checked', AsyncMock(return_value='image')):
            with self.assertRaisesRegex(OSError, 'heartbeat write failed'):
                await asyncio.wait_for(self.service.run(), 3)
        self.assertTrue(self.service.stopping.is_set())
        self.assertEqual(json.loads((self.queue/'service.json').read_text())['state'], 'stopped')

if __name__ == '__main__': unittest.main()
