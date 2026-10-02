"""Production contracts: train the declared cohort and reject changed inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import soundfile as sf
import torch

from rvcbench.benchmark.enkidu_stage import protect_enkidu


@pytest.mark.parametrize('mutate_source', [False, True])
def test_enkidu_full_cohort_and_output_lineage(tmp_path, mutate_source):
    root, model, output = tmp_path / 'data', tmp_path / 'model', tmp_path / 'out'
    root.mkdir()
    model.mkdir()
    rows = []
    for index in range(2):
        sf.write(root / f'{index}.wav', torch.zeros(2048).numpy(), 16000, subtype='PCM_16')
        rows.append(dict(dataset_name='LibriTTS', split='default', pair_id=str(index),
                         speaker_id='one', manifest_variant='speaker',
                         prompt_file_name=f'{index}.wav', target_file_name=f'{index}.wav',
                         prompt_text='hello', target_text='hello'))
    pd.DataFrame(rows).to_parquet(root / 'metadata.parquet')
    subset = tmp_path / 'subset.json'
    subset.write_text(json.dumps(rows[:1]))
    for name in ['hyperparams.yaml', 'embedding_model.ckpt', 'classifier.ckpt',
                 'mean_var_norm_emb.ckpt', 'label_encoder.ckpt']:
        (model / name).write_text('fixture')
    observed = []
    class Engine:
        def __init__(self, model_config, **kwargs):
            self.__dict__.update(kwargs)
            self.model = SimpleNamespace(parameters=lambda: [], eval=lambda: None)
            self.noises = {'one': {'real': torch.zeros(1), 'imag': torch.zeros(1)}}
        def generate_perturbations(self):
            for sid, batches in self.speaker_data.speaker_dataloaders.items():
                for index, batch in enumerate(batches):
                    observed.append(Path(batch.path_out[0]).name)
                    self.progress_callback(sid, 0, index, {})
            self.output_dir.mkdir(parents=True)
            torch.save(self.noises, self.output_dir / 'enkidu.noise')
        def save_protected_audio(self):
            for sid, batches in self.speaker_data.speaker_dataloaders.items():
                (self.output_dir / sid).mkdir()
                for batch in batches:
                    sf.write(self.output_dir / sid / Path(batch.path_out[0]).name,
                             batch.wav.flatten().numpy(), 16000, subtype='PCM_16')
            if mutate_source:
                sf.write(root / '0.wav', torch.ones(2048).numpy(), 16000, subtype='PCM_16')
    with patch('rvcbench.protection.enkidu.EnkiduProtector', Engine):
        if mutate_source:
            with pytest.raises(ValueError, match='inputs changed'):
                protect_enkidu(root, subset, model, output, epochs=1)
        else:
            result = protect_enkidu(root, subset, model, output, epochs=1)
            assert result['training_verified_steps'] == result['training_requested_steps'] == 2
            assert result['requested'] == result['verified'] == 1
    result = json.loads((output / 'stage_manifest.json').read_text())
    assert observed == ['0.wav', '1.wav']
    assert result['status'] == ('failed' if mutate_source else 'complete')
