"""minGRU: parallel log-space scan == sequential steps (with episode starts), finite gradients."""
import pytest
import torch

from conftest import needs_cuda
from pokerl.mingru import MinGRU
from pokerl.model import Policy, policy_kwargs

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make(dim, device, seed=0):
    torch.manual_seed(seed)
    m = MinGRU(dim).to(device)
    torch.nn.init.orthogonal_(m.proj.weight, 2 ** 0.5)  # the model's init
    torch.nn.init.zeros_(m.proj.bias)
    return m


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T,p_done", [(1, 0.0), (5, 0.3), (64, 0.05), (100, 0.05), (256, 0.02), (128, 0.0)])
def test_scan_equals_sequential(device, T, p_done):
    B, D = 16, 512
    m = make(D, device, T)
    g = torch.Generator(device="cpu").manual_seed(T)
    x = torch.randn(T, B, D, generator=g).to(device)
    h0 = torch.randn(B, D, generator=g).to(device)
    done = (torch.rand(T, B, generator=g) < p_done).float().to(device)
    done[0, :2] = 1.0  # always some reset at step 0
    with torch.no_grad():
        a = m.scan(x, h0, done)
        b = m.scan_sequential(x, h0, done)
    assert a.shape == (T, B, D)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_is_exact(device):
    """After a done the state no longer depends on anything before it."""
    T, B, D = 40, 4, 64
    m = make(D, device)
    x = torch.randn(T, B, D, device=device)
    done = torch.zeros(T, B, device=device)
    done[17] = 1
    a = m.scan(x, torch.randn(B, D, device=device) * 100, done)
    x2 = x.clone()
    x2[:17] = torch.randn(17, B, D, device=device) * 100
    b = m.scan(x2, torch.zeros(B, D, device=device), done)
    torch.testing.assert_close(a[17:], b[17:], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_gradients_finite_and_match(device):
    T, B, D = 50, 4, 64
    m = make(D, device)
    x = torch.randn(T, B, D, device=device, requires_grad=True)
    h0 = torch.randn(B, D, device=device, requires_grad=True)
    done = (torch.rand(T, B, device=device) < 0.1).float()
    w = torch.randn(T, B, D, device=device)
    grads = []
    for f in (m.scan, m.scan_sequential):
        m.zero_grad()
        x.grad = h0.grad = None
        (f(x, h0, done) * w).sum().backward()
        gs = [x.grad.clone(), h0.grad.clone()] + [p.grad.clone() for p in m.parameters()]
        assert all(torch.isfinite(t).all() for t in gs)
        grads.append(gs)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)


@needs_cuda
def test_extreme_gates_finite():
    m = make(64, "cuda")
    x = torch.randn(256, 8, 64, device="cuda") * 50  # saturated gates, large candidates
    x.requires_grad_(True)
    done = (torch.rand(256, 8, device="cuda") < 0.05).float()
    y = m.scan(x, torch.zeros(8, 64, device="cuda"), done)
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


@needs_cuda
def test_scan_under_bf16_autocast():
    """The update runs the scan under bf16 autocast (ppo.py): only the projection is bf16; softplus/log/exp/
    cumsum run in fp32, so the scan matches the fp32 recurrence on the same (bf16-rounded) projections to
    1e-4 (measured 1.4e-5), and the whole layer the fp32 layer within bf16 rounding (measured max 1.7e-2,
    mean 1.5e-3 at unit scale)."""
    T, B, D = 128, 64, 256
    m = make(D, "cuda")
    g = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(T, B, D, generator=g).cuda()
    h0 = torch.randn(B, D, generator=g).cuda()
    done = (torch.rand(T, B, generator=g) < 0.02).float().cuda()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m.scan(x, h0, done)
            k, ht = m.proj(x).chunk(2, dim=-1)
        assert y.dtype == torch.float32 and torch.isfinite(y).all()
        z, ht = torch.sigmoid(k.float()), ht.float()
        h, out = h0, []
        for t in range(T):
            h = (1 - z[t]) * (1 - done[t])[:, None] * h + z[t] * ht[t]
            out.append(h)
        torch.testing.assert_close(y, torch.stack(out), rtol=1e-4, atol=1e-4)
        ref = m.scan_sequential(x, h0, done)
    err = (y - ref).abs()
    assert err.max() < 0.05 and err.mean() < 5e-3


# ---- injection (rewind) ----------------------------------------------------------------------------------------

def _injections(T, B, D, device, seed):
    """Random dones and injections: several injections per sequence (some with done = 1, some without), the last
    two sequences without any; injected states at the scale of real states."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    done = (torch.rand(T, B, generator=g) < 0.05).float()
    done[0, :2] = 1.0
    inj = (torch.rand(T, B, generator=g) < 0.08).float()
    inj[:, -2:] = 0.0
    both = (torch.rand(T, B, generator=g) < 0.5) & (inj > 0)
    done[both] = 1.0  # a rewind: done and injection together
    inj[T // 2, 0], done[T // 2, 0] = 1.0, 1.0  # (at least one of each kind)
    inj[T - 1, 1], done[T - 1, 1] = 1.0, 0.0
    h_inj = torch.randn(T, B, D, generator=g) * (inj[..., None] > 0)
    assert ((inj > 0) & (done > 0)).any() and ((inj > 0) & (done == 0)).any()
    return done.to(device), inj.to(device), h_inj.to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T", [1, 5, 64, 100, 128])
def test_scan_with_injection(device, T):
    """scan == scan_sequential == a loop of step with injected states; the injection replaces the carried state."""
    B, D = 16, 256
    m = make(D, device, T)
    g = torch.Generator(device="cpu").manual_seed(100 + T)
    x = torch.randn(T, B, D, generator=g).to(device)
    h0 = torch.randn(B, D, generator=g).to(device)
    done, inj, h_inj = _injections(T, B, D, device, T)
    with torch.no_grad():
        a = m.scan(x, h0, done, inj, h_inj)
        b = m.scan_sequential(x, h0, done, inj, h_inj)
        h, out = h0, []
        for t in range(T):
            h = m.step(x[t], h, done[t], inj[t], h_inj[t])
            out.append(h)
        c = torch.stack(out)
        # reference formula: h_t = (1 - z)(1 - done) h_{t-1} + (1 - z) inj h_inj + z h~
        k, ht = m.proj(x).chunk(2, dim=-1)
        z = torch.sigmoid(k)
        h, ref = h0, []
        for t in range(T):
            h = (1 - z[t]) * (1 - done[t])[:, None] * h + (1 - z[t]) * inj[t][:, None] * h_inj[t] + z[t] * ht[t]
            ref.append(h)
        ref = torch.stack(ref)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b, c, rtol=0, atol=0)
    torch.testing.assert_close(b, ref, rtol=1e-5, atol=1e-5)  # (batched vs per-step projection GEMMs)
    # a rewind (done = 1 and an injection) equals carrying the injected state without done
    t, i = map(int, torch.nonzero((inj > 0) & (done > 0))[0])
    with torch.no_grad():
        hp = b[t - 1, i:i + 1] if t > 0 else h0[i:i + 1]
        direct = m.step(x[t, i:i + 1], h_inj[t, i:i + 1], torch.zeros(1, device=device))
        rew = m.step(x[t, i:i + 1], hp, torch.ones(1, device=device), torch.ones(1, device=device), h_inj[t, i:i + 1])
    assert torch.equal(direct, rew)


@pytest.mark.parametrize("device", DEVICES)
def test_injection_all_zero_is_bit_identical(device):
    T, B, D = 64, 8, 128
    m = make(D, device, 7)
    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.randn(T, B, D, generator=g).to(device)
    h0 = torch.randn(B, D, generator=g).to(device)
    done = (torch.rand(T, B, generator=g) < 0.05).float().to(device)
    inj = torch.zeros(T, B, device=device)
    h_inj = torch.randn(T, B, D, generator=g).to(device)  # (ignored where inj = 0)
    with torch.no_grad():
        assert torch.equal(m.scan(x, h0, done), m.scan(x, h0, done, inj, h_inj))
        assert torch.equal(m.scan_sequential(x, h0, done), m.scan_sequential(x, h0, done, inj, h_inj))
        assert torch.equal(m.step(x[3], h0, done[3]), m.step(x[3], h0, done[3], inj[3], h_inj[3]))


@pytest.mark.parametrize("device", DEVICES)
def test_injection_gradients_finite_and_match(device):
    T, B, D = 50, 4, 64
    m = make(D, device)
    x = torch.randn(T, B, D, device=device, requires_grad=True)
    h0 = torch.randn(B, D, device=device, requires_grad=True)
    done, inj, h_inj = _injections(T, B, D, device, 3)
    w = torch.randn(T, B, D, device=device)
    grads = []
    for f in (m.scan, m.scan_sequential):
        m.zero_grad()
        x.grad = h0.grad = None
        (f(x, h0, done, inj, h_inj) * w).sum().backward()
        gs = [x.grad.clone(), h0.grad.clone()] + [p.grad.clone() for p in m.parameters()]
        assert all(torch.isfinite(t).all() for t in gs)
        grads.append(gs)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)


@needs_cuda
def test_injection_under_bf16_autocast():
    """The learner's setting: bf16 projection, fp32 scan with injections (cf. test_scan_under_bf16_autocast)."""
    T, B, D = 128, 64, 256
    m = make(D, "cuda")
    g = torch.Generator(device="cpu").manual_seed(4)
    x = torch.randn(T, B, D, generator=g).cuda()
    h0 = torch.randn(B, D, generator=g).cuda()
    done, inj, h_inj = _injections(T, B, D, "cuda", 4)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m.scan(x, h0, done, inj, h_inj)
            k, ht = m.proj(x).chunk(2, dim=-1)
        assert y.dtype == torch.float32 and torch.isfinite(y).all()
        z, ht = torch.sigmoid(k.float()), ht.float()
        h, out = h0, []
        for t in range(T):
            h = (1 - z[t]) * ((1 - done[t])[:, None] * h + inj[t][:, None] * h_inj[t]) + z[t] * ht[t]
            out.append(h)
        torch.testing.assert_close(y, torch.stack(out), rtol=1e-4, atol=1e-4)


def make_chrono(dim, device, tau_max=512.0, seed=0):
    m = make(dim, device, seed)
    m.chrono_init(tau_max)
    return m


def test_chrono_init_sets_only_the_gate_biases():
    D, tau_max = 4096, 512.0
    m, plain = make_chrono(D, "cpu", tau_max), make(D, "cpu")
    b, cand = m.proj.bias[:D].detach(), m.proj.bias[D:].detach()
    assert b.max() <= 0.0 and b.min() >= -torch.log(torch.tensor(tau_max))
    assert b.min() < -6.0 and b.max() > -0.1  # the range is covered: decay 0.5 up to 512 / 513
    assert abs(b.mean() + torch.log(torch.tensor(tau_max)) / 2) < 0.15  # log tau uniform
    assert torch.equal(cand, torch.zeros(D))
    assert torch.equal(m.proj.weight, plain.proj.weight)
    half_life = -torch.log(torch.tensor(2.0)) / torch.log(1 - torch.sigmoid(b.double()))
    assert half_life.min() >= 0.99 and 300 < half_life.max() < 356


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T,p_done", [(64, 0.05), (256, 0.02), (128, 0.0)])
def test_chrono_scan_equals_sequential(device, T, p_done):
    B, D = 16, 512
    m = make_chrono(D, device, seed=T)
    g = torch.Generator(device="cpu").manual_seed(T)
    x = torch.randn(T, B, D, generator=g).to(device)
    h0 = torch.randn(B, D, generator=g).to(device)
    done = (torch.rand(T, B, generator=g) < p_done).float().to(device)
    done[0, :2] = 1.0
    with torch.no_grad():
        torch.testing.assert_close(m.scan(x, h0, done), m.scan_sequential(x, h0, done), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_chrono_done_still_kills_the_state(device):
    """Slow channels too: after a done the state no longer depends on anything before it. Exact in ``step``; in ``scan``
    fp32 rounding at these inputs of size 100 (measured over 20 seeds: at most 7e-5 with biases 0, 6e-5 with the chrono init)."""
    T, B, D = 40, 4, 64
    m = make_chrono(D, device)
    x = torch.randn(T, B, D, device=device)
    done = torch.zeros(T, B, device=device)
    done[17] = 1
    h0 = torch.randn(B, D, device=device) * 100
    x2 = x.clone()
    x2[:17] = torch.randn(17, B, D, device=device) * 100
    zero = torch.zeros(B, D, device=device)
    assert torch.equal(m.scan_sequential(x, h0, done)[17:], m.scan_sequential(x2, zero, done)[17:])
    torch.testing.assert_close(m.scan(x, h0, done)[17:], m.scan(x2, zero, done)[17:], rtol=1e-3, atol=1e-4)


def test_chrono_policy_off_by_default_and_from_the_config():
    kw = dict(pixel_feat=512, verbose=False)
    torch.manual_seed(3)
    off = Policy(**kw).state_dict()
    torch.manual_seed(3)
    same = Policy(chrono_tau=0.0, **kw).state_dict()
    assert all(torch.equal(off[k], same[k]) for k in off)  # (no random number drawn when off)
    assert all((off[f"grus.{i}.proj.bias"] == 0).all() for i in range(2))
    assert policy_kwargs({})["chrono_tau"] == 0.0
    c = policy_kwargs({"pixel_feat": 512, "chrono_tau": 512.0})
    assert c["chrono_tau"] == 512.0
    torch.manual_seed(3)
    m = Policy(verbose=False, **c)
    gates = [m.grus[i].proj.bias[:512].detach() for i in range(2)]
    assert all(g.max() <= 0 and g.min() < -6.0 for g in gates)
    assert not torch.equal(gates[0], gates[1])  # (one draw per layer)
    assert all((m.grus[i].proj.bias[512:] == 0).all() for i in range(2))
    assert torch.equal(m.pi.bias, torch.zeros_like(m.pi.bias)) and torch.equal(m.vf.bias, torch.zeros_like(m.vf.bias))
