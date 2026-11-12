from dataclasses import replace

import torch

from energy import EnergyProvider
from train import train_fixed_budget


def test_explicit_validation_schedule_and_final_validation(smoke_config, tmp_path):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, tokens, targets):
            return None, (self.weight * tokens.float() - targets.float()).square().mean()

    profile = replace(smoke_config, budget=replace(
        smoke_config.budget, max_optimizer_steps=1000, grad_accum_steps=1,
        evaluation_interval=100, validation_at_steps=(300, 650),
    ))
    batch = (torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long))
    result = train_fixed_budget(
        model=TinyModel(), train_dataloader=[batch], validation_dataloader=[batch],
        profile=profile, device=torch.device('cpu'), run_dir=str(tmp_path),
        energy_provider=EnergyProvider.create(torch.device('cpu')),
    )
    assert [entry['step'] for entry in result['validation_history']] == [300, 650]
    assert result['training']['fixed_optimizer_steps'] == 1000
    assert result['final_validation']['steps'] == 1
