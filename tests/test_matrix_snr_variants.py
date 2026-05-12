import torch
import torch.nn as nn

from snr_grad import RotatedSNRAdamW, SpectralSNRMuon


def _run_steps(model, opt, steps=3):
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()


def test_rotated_snr_adamw_runs_and_updates_matrix_weight():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 1))
    before = model[0].weight.detach().clone()
    opt = RotatedSNRAdamW(model.parameters(), lr=1e-3, basis_update_interval=1)
    _run_steps(model, opt)
    assert not torch.equal(before, model[0].weight.detach())


def test_spectral_snr_muon_diag_and_full_run():
    torch.manual_seed(0)
    for mode in ("diag", "full"):
        model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 1))
        before = model[0].weight.detach().clone()
        opt = SpectralSNRMuon(model.parameters(), lr=1e-3, mode=mode, variant="adam_spectral_gate")
        _run_steps(model, opt)
        assert not torch.equal(before, model[0].weight.detach())
