"""Tests for the AP/GP operation-order wrapper in benchmark_dopr_order.py.

These validate the experimental ``OrderedDoPr`` wrapper used to compare the
shipped AP->GP order against the swapped GP->AP order:

* When no layer is registered for activation preconditioning, every mode reduces
  to the bare base optimizer step (AP is a no-op), so all modes must move the
  weights identically.
* ``post_norm`` must preserve the base GP update's per-parameter norm.
* ``post`` must realize ``W <- W - eta * (D @ S_z^-1)`` exactly (AP applied to the
  update direction, not the gradient).
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snr_grad import ActivationPrecondConfig, SNRAdamW
from benchmark_dopr_order import OrderedDoPr


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(6, 4, bias=False), nn.Linear(4, 3, bias=False))
    return m


def _one_step(model, opt, x, target):
    opt.zero_grad()
    loss = ((model(x) - target) ** 2).sum(dim=1).mean()
    loss.backward()
    opt.step()


def test_no_registered_layers_all_modes_match_base():
    """Excluding every layer makes AP a no-op, so all modes match the base step."""
    x = torch.randn(8, 6)
    target = torch.randn(8, 3)
    cfg = ApExcludeAll()

    final = {}
    for mode in ("off", "pre", "post", "post_norm"):
        model = _tiny_model(seed=1)
        base = SNRAdamW(model.parameters(), lr=1e-2)
        opt = OrderedDoPr(base, model, cfg, mode)
        _one_step(model, opt, x, target)
        final[mode] = [p.detach().clone() for p in model.parameters()]

    ref = final["off"]
    for mode in ("pre", "post", "post_norm"):
        for a, b in zip(final[mode], ref):
            assert torch.allclose(a, b, atol=1e-6), f"mode {mode} diverged from base"


def ApExcludeAll():
    # Exclude both linear layers ("0" and "1") so nothing is preconditioned.
    return ActivationPrecondConfig(damping=0.1, exclude_modules=["0", "1"])


def test_post_norm_preserves_base_update_norm():
    """post_norm rescales each conditioned update back to the base GP update norm."""
    x = torch.randn(8, 6)
    target = torch.randn(8, 3)
    cfg = ActivationPrecondConfig(damping=0.1)

    # Bare base step (off) to get the reference per-parameter update norms.
    m_off = _tiny_model(seed=2)
    base_off = SNRAdamW(m_off.parameters(), lr=1e-2)
    p0_off = [p.detach().clone() for p in m_off.parameters()]
    _one_step(m_off, OrderedDoPr(base_off, m_off, cfg, "off"), x, target)
    ref_norms = [float((p.detach() - p0).norm()) for p, p0 in zip(m_off.parameters(), p0_off)]

    # post_norm step from an identical init / identical batch.
    m_pn = _tiny_model(seed=2)
    base_pn = SNRAdamW(m_pn.parameters(), lr=1e-2)
    p0_pn = [p.detach().clone() for p in m_pn.parameters()]
    _one_step(m_pn, OrderedDoPr(base_pn, m_pn, cfg, "post_norm"), x, target)
    pn_norms = [float((p.detach() - p0).norm()) for p, p0 in zip(m_pn.parameters(), p0_pn)]

    for a, b in zip(pn_norms, ref_norms):
        assert abs(a - b) <= 1e-5 + 1e-4 * b, f"post_norm changed update norm: {a} vs {b}"


def test_post_applies_ap_to_update_direction():
    """post realizes W <- p0 + (delta @ S_d^-1), with delta the base GP update."""
    torch.manual_seed(3)
    x = torch.randn(10, 6)
    target = torch.randn(10, 3)
    cfg = ActivationPrecondConfig(damping=0.1)

    model = _tiny_model(seed=3)
    base = SNRAdamW(model.parameters(), lr=1e-2)
    opt = OrderedDoPr(base, model, cfg, "post")

    # Capture the base GP update (delta) by snapshotting around a bare base step on
    # a clone with identical state, fed the identical batch.
    model_ref = _tiny_model(seed=3)
    base_ref = SNRAdamW(model_ref.parameters(), lr=1e-2)
    p0_ref = [p.detach().clone() for p in model_ref.parameters()]
    _one_step(model_ref, OrderedDoPr(base_ref, model_ref, cfg, "off"), x, target)
    deltas = [p.detach() - p0 for p, p0 in zip(model_ref.parameters(), p0_ref)]

    # Activations under the *initial* weights (the forward that produced the grads):
    # layer 0 sees x, layer 1 sees layer0(x).
    m_init = _tiny_model(seed=3)
    with torch.no_grad():
        a0 = x
        a1 = m_init[0](x)

    p0_post = [p.detach().clone() for p in model.parameters()]
    _one_step(model, opt, x, target)

    def damped_inv_solve(delta, z):
        z = z.reshape(-1, z.shape[-1]).double()
        n = z.shape[0]
        S = (z.t() @ z) / n
        d_z = S.shape[0]
        tau = 0.1 * (S.diagonal().sum() / d_z) + 1e-8
        Sd = S + tau * torch.eye(d_z, dtype=torch.double)
        return (delta.double() @ torch.linalg.inv(Sd)).float()

    expected_w0 = p0_post[0] + damped_inv_solve(deltas[0], a0)
    expected_w1 = p0_post[1] + damped_inv_solve(deltas[1], a1)

    got_w0, got_w1 = [p.detach() for p in model.parameters()]
    assert torch.allclose(got_w0, expected_w0, atol=1e-4), "layer0 post update mismatch"
    assert torch.allclose(got_w1, expected_w1, atol=1e-4), "layer1 post update mismatch"
