"""The policy: shapes, the v76 parameter count, step and sequence agreeing, the rewind injection, and (with
$POKERL_V76_CKPT pointing at a checkpoint of the v76 run) loading the v76 network."""
import os

import pytest
import torch

from pokerl.model import NO_ACTION, Policy, num_params, policy_kwargs
from pokerl.train import TrainConfig

V76_PARAMS = 2_694_793


def obs(T, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"pixels": torch.randint(0, 256, (T, B, 80, 120), dtype=torch.uint8, generator=g),
            "trail": torch.randint(0, 256, (T, B, 11, 15), dtype=torch.uint8, generator=g),
            "prev_action": torch.randint(0, NO_ACTION + 1, (T, B), generator=g)}


def test_v76_shape():
    torch.manual_seed(0)
    m = Policy(**policy_kwargs(TrainConfig().to_json()), verbose=False)
    assert num_params(m) == V76_PARAMS
    assert m.pixel_shape == (80, 120) and m.hidden == 512 and m.layers == 2


def test_step_equals_sequence():
    torch.manual_seed(0)
    m = Policy(hidden=64, pixel_feat=32, chrono_tau=16, verbose=False)
    T, B = 12, 3
    o = obs(T, B)
    done = torch.zeros(T, B)
    done[0] = 1
    done[5, 1] = 1
    mask = torch.ones(T, B, 8, dtype=torch.bool)
    mask[:, :, 3] = False
    h0 = m.initial_state(B)
    with torch.no_grad():
        logits, values = m.sequence(o, h0, done, mask)
        h = h0
        for t in range(T):
            lg, v, h = m.step({k: x[t] for k, x in o.items()}, h, done[t], mask[t])
            assert torch.allclose(lg, logits[t], atol=1e-4) and torch.allclose(v, values[t], atol=1e-4)
    assert (logits[:, :, 3] < -1e8).all()


def test_injection_replaces_the_state():
    torch.manual_seed(0)
    m = Policy(hidden=32, pixel_feat=16, verbose=False)
    o = {k: x[0] for k, x in obs(1, 2).items()}
    mask = torch.ones(2, 8, dtype=torch.bool)
    h_inj = torch.randn(2, 2, 32)
    with torch.no_grad():
        a = m.step(o, torch.randn(2, 2, 32), torch.ones(2), mask, torch.ones(2), h_inj)
        b = m.step(o, torch.zeros(2, 2, 32), torch.ones(2), mask, torch.ones(2), h_inj)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[2], b[2])  # (done kills the carried state)


@pytest.mark.skipif(not os.environ.get("POKERL_V76_CKPT"), reason="$POKERL_V76_CKPT: a v76 checkpoint")
def test_loads_the_v76_checkpoint():
    ck = torch.load(os.environ["POKERL_V76_CKPT"], map_location="cpu", weights_only=False)
    m = Policy(**policy_kwargs(ck["config"]), verbose=False)
    m.load_state_dict(ck["model"])
    assert num_params(m) == V76_PARAMS
