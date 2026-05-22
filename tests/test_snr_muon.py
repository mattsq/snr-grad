import torch
import torch.nn as nn

from snr_grad import SNRMuon


def test_snr_muon_runs_pre_and_post_modes():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 1))
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)

    for mode in ("pre", "post"):
        opt = SNRMuon(model.parameters(), lr=1e-3, muon_mode=mode)
        for _ in range(3):
            opt.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()


def test_snr_muon_updates_2d_weight():
    torch.manual_seed(0)
    layer = nn.Linear(4, 4, bias=False)
    before = layer.weight.detach().clone()
    opt = SNRMuon(layer.parameters(), lr=1e-2)

    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    opt.zero_grad()
    loss = ((layer(x) - y) ** 2).mean()
    loss.backward()
    opt.step()

    assert not torch.equal(before, layer.weight.detach())


def test_snr_muon_runs_with_grokfast_enabled():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.ReLU(), nn.Linear(8, 1))
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)
    opt = SNRMuon(model.parameters(), lr=1e-3, grokfast_alpha=0.9, grokfast_lamb=2.0)
    for _ in range(3):
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    param = next(model.parameters())
    state = opt.state[param]
    assert "g_slow" in state
    assert state["step"] == 3
