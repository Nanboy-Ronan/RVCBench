import atexit
import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from rvcbench.models.worker_protocol import WorkerResponseReader


@pytest.fixture
def worker():
    started = []

    def start(code):
        proc = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, bufsize=1)
        started.append(proc)
        return proc

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except OSError:
                pass


def read(proc, reader=None, event='response', timeout=20):
    reader = reader or WorkerResponseReader('Test')
    return reader.read(proc, expect_event=event, timeout_sec=timeout, logger=None,
                       stderr_tail=lambda: 'recorded stderr tail')


def emit(*lines, then=''):
    payload = ''.join(line + '\n' for line in lines)
    return f'import sys, time; sys.stdout.write({payload!r}); sys.stdout.flush(); {then}'


def test_reader_skips_blank_and_non_json_stdout(worker):
    proc = worker(emit('loading weights', '', '{"event": "response", "ok": true}'))
    assert read(proc) == {'event': 'response', 'ok': True}


def test_reader_returns_buffered_messages_in_order(worker):
    proc = worker(emit('{"event": "response", "n": 1}', '{"event": "response", "n": 2}'))
    reader = WorkerResponseReader('Test')
    assert [read(proc, reader)['n'], read(proc, reader)['n']] == [1, 2]


def test_reader_passes_startup_error_to_the_caller(worker):
    proc = worker(emit('{"event": "startup_error", "ok": false, "error": "no weights"}'))
    assert read(proc, event='ready')['error'] == 'no weights'


def test_reader_rejects_unexpected_event(worker):
    proc = worker(emit('{"event": "ready", "ok": true}'))
    with pytest.raises(RuntimeError, match='unexpected protocol event'):
        read(proc)


def test_reader_rejects_non_object_json(worker):
    proc = worker(emit('[1, 2]'))
    with pytest.raises(RuntimeError, match='requires JSON objects'):
        read(proc)


def test_reader_times_out_on_silent_worker(worker):
    proc = worker('import time; time.sleep(60)')
    with pytest.raises(TimeoutError, match='response timed out after 0.2s'):
        read(proc, timeout=0.2)


def test_reader_does_not_wait_for_a_terminator_past_the_deadline(worker):
    proc = worker('import sys, time; sys.stdout.write(\'{"event": "response"\'); sys.stdout.flush(); time.sleep(60)')
    with pytest.raises(TimeoutError):
        read(proc, timeout=0.5)


def test_reader_reports_closed_stdout_with_stderr_tail(worker):
    proc = worker('raise SystemExit(3)')
    with pytest.raises(RuntimeError, match=r'stdout closed \(exit=.*recorded stderr tail'):
        read(proc)


def test_reader_bounds_unterminated_output(worker):
    proc = worker("import sys, time; sys.stdout.write('x' * (5 * 1024 * 1024)); sys.stdout.flush(); time.sleep(60)")
    with pytest.raises(RuntimeError, match='protocol buffer limit'):
        read(proc)


def test_reader_requires_a_running_worker():
    with pytest.raises(RuntimeError, match='worker is not running'):
        read(None)


REPLY = ('import json, sys, time\n'
         'request = json.loads(sys.stdin.readline())\n'
         'def reply(**fields):\n'
         '    print(json.dumps(dict(event="response", **fields)), flush=True)\n')
BEHAVIOURS = {
    'mismatched_request': ('reply(ok=True, request_id="another-request", seed=request["seed"])',
                           RuntimeError, 'does not match the request'),
    'worker_error': ('reply(ok=False, request_id=request["request_id"], error="decoder failed")',
                     RuntimeError, 'decoder failed'),
    'unacknowledged_seed': ('reply(ok=True, request_id=request["request_id"], seed=request["seed"] + 1)',
                            RuntimeError, 'did not acknowledge the requested seed'),
    'boolean_seed': ('reply(ok=True, request_id=request["request_id"], seed=True)',
                     RuntimeError, 'did not acknowledge the requested seed'),
    'other_output_path': ('reply(ok=True, request_id=request["request_id"], seed=request["seed"], '
                          'output_path=request["output_path"] + ".other")',
                          RuntimeError, 'unexpected output path'),
    'silent': ('time.sleep(60)', TimeoutError, 'timed out'),
}
GENERATORS = {
    'maskgct': ('MaskGCTGenerator', lambda ref, out: dict(
        prompt_speech_path=ref, prompt_text='reference words', target_text='target words', output_path=out, seed=7)),
    'index_tts': ('IndexTTSGenerator', lambda ref, out: dict(ref_audio=ref, text='target words', output_path=out, seed=7)),
}


@pytest.mark.parametrize('module_name', sorted(GENERATORS))
@pytest.mark.parametrize('behaviour', sorted(BEHAVIOURS))
def test_worker_generators_fail_closed_and_reap_the_worker(tmp_path, worker, module_name, behaviour):
    class_name, arguments = GENERATORS[module_name]
    cls = getattr(importlib.import_module(f'rvcbench.models.{module_name}.generator'), class_name)
    action, error, message = BEHAVIOURS[behaviour]
    generator = cls.__new__(cls)
    process = worker(REPLY + action)
    generator._process, generator._stderr_handle, generator._stderr_path = process, None, None
    generator._response_reader = WorkerResponseReader(class_name)
    generator.logger = None
    generator.config = SimpleNamespace(request_timeout_sec=0.5 if behaviour == 'silent' else 20,
                                       output_wait_timeout_sec=0, output_wait_poll_interval_sec=0.01)
    generator.ensure_model = lambda: None
    generator.last_native_seed = generator.last_native_requested_seed = 7
    reference = tmp_path / 'reference.wav'
    reference.write_bytes(b'reference')
    try:
        with pytest.raises(error, match=message):
            generator.generate(**arguments(reference, tmp_path / 'out' / 'cloned.wav'))
        assert process.poll() is not None
        assert generator._process is None
        assert generator.last_native_seed is None and generator.last_native_requested_seed is None
        assert not (tmp_path / 'out' / 'cloned.wav').exists()
    finally:
        atexit.unregister(generator.close)
