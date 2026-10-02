"""Bounded JSONL response reading for owned, serial inference workers."""
import json
import os
import selectors
import time


class WorkerResponseReader:
    def __init__(self, name):
        self.name = name
        self.buffer = bytearray()

    def clear(self):
        self.buffer.clear()

    def read(self, proc, *, expect_event, timeout_sec, logger, stderr_tail):
        if proc is None or proc.stdout is None:
            raise RuntimeError(f"{self.name} worker is not running.")
        deadline = time.monotonic() + float(timeout_sec)
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                while b'\n' in self.buffer:
                    raw, _, remaining = self.buffer.partition(b'\n')
                    self.buffer = bytearray(remaining)
                    if not raw.strip():
                        continue
                    try:
                        message = json.loads(raw)
                    except (ValueError, UnicodeDecodeError):
                        if logger:
                            logger.debug('[%s] Ignoring non-JSON worker stdout: %r', self.name, raw[:500])
                        continue
                    if not isinstance(message, dict):
                        raise RuntimeError(f'{self.name} worker protocol requires JSON objects')
                    if message.get('event') not in (expect_event, 'startup_error'):
                        raise RuntimeError(f'{self.name} worker returned an unexpected protocol event')
                    return message
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(f'{self.name} worker {expect_event} timed out after {timeout_sec}s')
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f'{self.name} worker stdout closed (exit={proc.poll()}). {stderr_tail()}')
                self.buffer.extend(chunk)
                if len(self.buffer) > 4 * 1024 * 1024:
                    raise RuntimeError(f'{self.name} worker stdout exceeded the protocol buffer limit')
