"""A checkpoint error must not leave partially changed model weights."""
import logging

import pytest
import torch

from rvcbench.utils.commons import load_checkpoint


@pytest.mark.parametrize('issue', ['missing', 'shape', 'unexpected'])
def test_strict_checkpoint_rejects_incomplete_weights_without_mutation(tmp_path, issue):
    model = torch.nn.Linear(3, 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    saved = {k: torch.ones_like(v) for k, v in before.items()}
    if issue == 'missing':
        saved.pop('bias')
    elif issue == 'shape':
        saved['weight'] = torch.ones(2, 4)
    else:
        saved['other'] = torch.ones(1)
    path = tmp_path / 'model.pt'
    torch.save({'model': saved}, path)
    with pytest.raises(ValueError, match='incompatible'):
        load_checkpoint(path, model, logging.getLogger())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_partial_checkpoint_requires_opt_in_and_exposes_retained_initialization(tmp_path):
    model = torch.nn.Linear(3, 2)
    bias = model.bias.detach().clone()
    path = tmp_path / 'model.pt'
    torch.save({'model': {'weight': torch.ones_like(model.weight)}, 'iteration': 12}, path)
    result = load_checkpoint(path, model, logging.getLogger(), strict=False)
    assert result[-1] == 12 and torch.equal(model.bias, bias)
    assert torch.equal(model.weight, torch.ones_like(model.weight))
    assert model._checkpoint_load_report['policy'] == 'legacy_partial'
    assert model._checkpoint_load_report['missing'] == ['bias']
    assert model._checkpoint_load_report['loaded_tensors'] == 1


def test_complete_checkpoint_records_full_loading(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    load_checkpoint(path, model, logging.getLogger())
    assert model._checkpoint_load_report['loaded_tensors'] == 2
    assert not model._checkpoint_load_report['missing']
