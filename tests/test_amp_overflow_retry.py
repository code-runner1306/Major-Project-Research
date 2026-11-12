from dataclasses import replace

import pytest
import torch

from energy import EnergyProvider
from train import train_fixed_budget


@pytest.mark.parametrize('always_overflow', [False, True])
def test_scaled_gradient_overflow_retries_same_optimizer_step(monkeypatch, smoke_config, tmp_path, always_overflow):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        def forward(self, tokens, targets):
            self.calls += 1
            loss = (self.weight * tokens.float() - targets.float()).square().mean()
            if always_overflow or self.calls == 1:
                loss.register_hook(lambda gradient: gradient * float('inf'))
            return None, loss

    real_scaler = torch.amp.GradScaler
    monkeypatch.setattr(torch.amp, 'GradScaler', lambda *_args, **_kwargs: real_scaler('cpu', init_scale=65536))
    model = Model()
    profile = replace(smoke_config, budget=replace(smoke_config.budget, max_optimizer_steps=2, grad_accum_steps=1))
    batch = (torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long))
    args = dict(model=model, train_dataloader=[batch], validation_dataloader=[batch], profile=profile,
                device=torch.device('cpu'), run_dir=str(tmp_path), energy_provider=EnergyProvider.create(torch.device('cpu')))
    if always_overflow:
        with pytest.raises(FloatingPointError, match='optimizer step 1'):
            train_fixed_budget(**args)
        assert model.calls == 17
    else:
        result = train_fixed_budget(**args)
        assert result['training']['fixed_optimizer_steps'] == 2
        assert result['training']['processed_tokens'] == 2
        assert result['training']['stability_events'] == [
            {'step': 1, 'kind': 'amp_overflow_retry', 'scale': 65536.0, 'new_scale': 32768.0}
        ]
