"""CPU-only timing evidence and publication regressions; no model loading."""
from pathlib import Path
import unittest

from src.benchmark.timing import timing_summary, compare_timing

ROOT = Path(__file__).resolve().parents[1]
QWEN = 'qwen3_generate_excluding_prompt_encoding_and_io_v1'
F5 = 'f5_sequential_inference_excluding_transcription_and_output_write_v1'


def row(elapsed=2, duration=4, scope=QWEN):
    return dict(synthesis_time_sec=elapsed, generated_duration_sec=duration, timing_scope=scope)


class TimingTests(unittest.TestCase):
    def test_raw_ratio_is_duration_weighted_and_never_authorizes_ranking(self):
        result = timing_summary([row(), row(1, 1)])
        self.assertEqual(result['rtf'], 3 / 5)
        self.assertEqual(result['rtf_by_scope'][QWEN]['samples'], 2)
        self.assertFalse(result['timing_comparability']['ranking_allowed'])

    def test_equal_legacy_labels_do_not_prove_equal_boundaries(self):
        for scope in ('adapter_reported_synthesis', 'adapter_call_including_lazy_initialization', QWEN, F5, 'invented_common_scope'):
            result = compare_timing([row(scope=scope)], [row(scope=scope)])
            self.assertEqual(result['status'], 'not_comparable')
            self.assertIsNone(result['rtf_delta'])
            self.assertFalse(result['ranking_allowed'])

    def test_mixed_scopes_are_preserved_without_pooled_rtf(self):
        result = timing_summary([row(), row(scope=F5)])
        self.assertIsNone(result['rtf'])
        self.assertEqual(set(result['rtf_by_scope']), {QWEN, F5})
        self.assertFalse(compare_timing([row()], [row(scope=F5)])['ranking_allowed'])

    def test_missing_nonfinite_negative_and_zero_duration(self):
        for bad in (row(None), row(float('nan')), row(float('inf')), row(-1), row(duration=0), row(duration=-1), row(duration=float('inf'))):
            result = timing_summary([row(), bad])
            self.assertIsNone(result['rtf'])
            self.assertEqual(result['timing_comparability']['invalid_samples'], 1)
        self.assertIsNone(timing_summary([row(scope=None)])['rtf'])
        self.assertFalse(timing_summary([])['timing_comparability']['ranking_allowed'])

    def test_historical_missing_scope_does_not_inherit_current_scope(self):
        result = compare_timing([row()], [row(scope=None)])
        self.assertEqual(result['right']['timing_comparability']['scopes'], ['unspecified'])
        self.assertIsNone(result['rtf_delta'])

    def test_legacy_csv_scope_roundtrip_without_optional_model_imports(self):
        # Compile the production reader alone: importing the legacy evaluation
        # module would require unrelated Whisper/Torch scoring dependencies.
        import ast
        import csv
        import tempfile
        module = ast.parse((ROOT / 'src/evaluation/generation.py').read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_load_synthesis_timings')
        namespace = {'Path': Path, 'csv': csv}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'reader', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            timing = path / 'synthesis_timings.csv'
            timing.write_text('generated_path,synthesis_time_sec,timing_scope\na.wav,2,' + F5 + '\n')
            records = namespace['_load_synthesis_timings'](path)
            self.assertEqual(records[str(Path('a.wav').resolve())]['timing_scope'], F5)
            timing.write_text('generated_path,synthesis_time_sec\na.wav,2\n')
            records = namespace['_load_synthesis_timings'](path)
            self.assertIsNone(records[str(Path('a.wav').resolve())]['timing_scope'])

    def test_export_keeps_raw_scope_and_adapter_call_time(self):
        import csv
        import tempfile
        import numpy as np
        import soundfile as sf
        from src.evaluation.pipeline import _export
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            audio = output / 'sample.wav'
            sf.write(audio, np.zeros(8000), 8000)
            sample = dict(row(2, 1), sample_id='a', speaker_id='s',
                          target_path=str(audio), target_text='hello',
                          generated_path=str(audio), generated_sha256='fixture',
                          adapter_call_time_sec=9, metrics={'sim': .5})
            result = _export([sample], ['sim'], output, {}, None)
            self.assertEqual(result['rtf'], 2)
            self.assertFalse(result['timing_comparability']['ranking_allowed'])
            with (output / 'generation_sample_metrics.csv').open() as stream:
                saved = next(csv.DictReader(stream))
            self.assertEqual(saved['timing_scope'], QWEN)
            self.assertEqual(saved['synthesis_time_sec'], '2')
            self.assertEqual(saved['adapter_call_time_sec'], '9')
            sample['timing_scope'] = F5
            missing = dict(sample, sample_id='b', generated_sha256=None)
            mixed = _export([sample, missing], ['sim'], output, {}, None)
            self.assertIsNone(mixed['rtf'])

    def test_site_has_no_rtf_sort_control_or_sort_data(self):
        for path in ('docs/site-src/template.html', 'docs/index.html'):
            source = (ROOT / path).read_text()
            self.assertNotIn('data-key="rtf"', source)
            self.assertIn('RTF (raw, incomparable)', source)
        self.assertNotIn('data-rtf=', (ROOT / 'docs/index.html').read_text())
        readme = (ROOT / 'README.md').read_text()
        self.assertIn('RTF (raw, incomparable)', readme)
        self.assertNotIn('**0.08**', readme)


if __name__ == '__main__':
    unittest.main()
