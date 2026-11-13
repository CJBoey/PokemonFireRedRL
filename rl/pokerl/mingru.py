"""minGRU layer (Feng et al. 2024, "Were RNNs All We Needed?").

    z_t = σ(W_z x_t),  h̃_t = W_h x_t,  h_t = a_t ⊙ h_{t-1} + z_t ⊙ h̃_t,  a_t = (1 − z_t)(1 − done_t)

``done_t`` = 1 when x_t is the first input of a new episode: the state is reset to 0 before step t.

**Injection** (rewind, env.py / ppo.py): an injected state ``h_inj_t`` (``inj_t`` = 1) enters as if it were
the carried state, ``h_t = (1 − z_t)·(1 − done_t)·h_{t−1} + (1 − z_t)·inj_t·h_inj_t + z_t·h̃_t`` (a rewind has
``done_t`` = 1 too: the carried state is killed and replaced). ``step``, ``scan`` and ``scan_sequential`` take the
optional ``inj`` [B] / [T, B] and ``h_inj`` [B, D] / [T, B, D]; without them the computation is unchanged. In
``scan_log`` the injected term goes into the signed ``b_t`` next to ``z_t h̃_t``: where ``inj_t`` = 1,
``b_t = z_t h̃_t + (1 − z_t) h_inj_t`` is formed in linear space and then split into its positive and negative parts
like ``z_t h̃_t`` (splitting each term separately would leave both parts nonzero when the signs differ, and the
final ``h⁺ − h⁻`` would cancel their scan errors: measured 1.4e-5 vs ≤ 1e-5); elsewhere ``b`` is bit-identical to
the code without injection; ``log a`` is unchanged (a rewind step, ``done`` = 1, still kills the carry).

**Chrono-style gate init** (:meth:`MinGRU.chrono_init`, ``--chrono-tau``): only the initial gate biases change,
``step`` and ``scan`` do not, and ``done`` still kills the state.

* :meth:`MinGRU.step` — one step (rollouts), fp32.
* :meth:`MinGRU.scan` — the whole sequence at once with the parallel scan in log space (Heinsen 2023,
  "Efficient Parallelization of a Ubiquitous Sequential Computation"):
  ``h_t = exp(A_t) · Σ_{s≤t} exp(log b_s − A_s)`` with ``A_t = Σ_{r≤t} log a_r`` (cumsum) and the sum as
  a ``logcumsumexp``; ``h_0`` enters as ``b_0`` with ``a_0 = 1``. Numerically safe form:
  ``log(1 − z) = −softplus(k)``, ``log z = −softplus(−k)`` for the gate logit ``k``.
  - Signed values: ``b_t = z_t h̃_t`` (and ``h_0``) are split into positive and negative parts, each
    scanned in log space (log of a part clamped at 1e-30), ``h = h⁺ − h⁻``.
  - Episode starts: ``log a_t = log(1 − z_t) − DONE_LOG`` (20: the carried state is scaled by
    exp(−20) ≈ 2e-9, i.e. reset to 0 below fp32 resolution) instead of −inf, so every quantity and
    gradient stays finite.
  - Precision: the final ``A_t + LSE_t`` cancels two numbers of size |A_t|, losing ~|A_t|·6e-8. The
    log-space scan therefore runs on chunks of ``CHUNK`` (8) steps, all chunks in parallel (|A| bounded
    by the chunk), and the chunks are joined by the exact linear carry
    ``h = h_local + exp(A_local) · h_prev`` (T/8 small elementwise steps on [B, D]). fp32 throughout;
    max |scan − sequential| ≈ 1e-5 or less at unit-scale inputs (rl/tests/test_mingru.py).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DONE_LOG = 20.0
CHUNK = 8
_TINY = 1e-30


def _log_parts(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(log max(v, tiny), log max(−v, tiny))."""
    return torch.log(torch.clamp(v, min=_TINY)), torch.log(torch.clamp(-v, min=_TINY))


def scan_log(log_a: torch.Tensor, log_z: torch.Tensor, h_tilde: torch.Tensor, h0: torch.Tensor,
             chunk: int = CHUNK, inj: Optional[torch.Tensor] = None, log_c: Optional[torch.Tensor] = None,
             h_inj: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Solves h_t = a_t h_{t-1} + z_t h̃_t for t = 0..T-1 given log a [T, B, D], log z [T, B, D],
    h̃ [T, B, D] and the state before step 0, h0 [B, D]. Returns h [T, B, D]. With ``inj`` [T, B] (nonzero =
    injection), ``log_c`` = log(1 − z) [T, B, D] and ``h_inj`` [T, B, D]: b_t gains (1 − z_t) h_inj_t where inj_t.

    All chunks are scanned in parallel (Heinsen, from a zero state, along the chunk axis); the T/chunk
    chunk carries are then chained exactly, and added back with one broadcast."""
    t_len = log_a.shape[0]
    rest = log_a.shape[1:]
    hp, hn = _log_parts(h_tilde)
    log_bp, log_bn = log_z + hp, log_z + hn
    if inj is not None:  # where injected: the signed b = z h~ + (1 - z) h_inj, split into its parts as above
        bp, bn = _log_parts(torch.exp(log_z) * h_tilde + torch.exp(log_c) * h_inj)
        m = (inj != 0)[..., None]
        log_bp = torch.where(m, bp, log_bp)
        log_bn = torch.where(m, bn, log_bn)
    nc = -(-t_len // chunk)
    pad = nc * chunk - t_len
    if pad:
        log_a = F.pad(log_a, (0, 0) * len(rest) + (0, pad))
        fill = torch.full((pad,) + rest, float(torch.log(torch.tensor(_TINY))), dtype=log_bp.dtype, device=log_bp.device)
        log_bp = torch.cat([log_bp, fill])
        log_bn = torch.cat([log_bn, fill])
    shape = (nc, chunk) + rest
    log_a, log_bp, log_bn = log_a.reshape(shape), log_bp.reshape(shape), log_bn.reshape(shape)
    a_star = torch.cumsum(log_a, dim=1)
    h_loc = torch.exp(a_star + torch.logcumsumexp(log_bp - a_star, dim=1)) - \
        torch.exp(a_star + torch.logcumsumexp(log_bn - a_star, dim=1))
    decay = torch.exp(a_star)  # product of a from the chunk start
    h_in = [h0]
    for c in range(nc - 1):
        h_in.append(h_loc[c, -1] + decay[c, -1] * h_in[-1])
    h = h_loc + decay * torch.stack(h_in)[:, None]
    return h.reshape((nc * chunk,) + rest)[:t_len]


class MinGRU(nn.Module):
    """minGRU over ``dim`` features (input size = hidden size = ``dim``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, 2 * dim)  # [gate logit k | candidate h̃]

    def chrono_init(self, tau_max: float) -> None:
        """Per channel, the gate logit's bias = −log τ with τ log-uniform in [1, ``tau_max``]: the resting carry
        1 − σ(b) = τ / (1 + τ) has a half-life of 1 step at τ = 1 and about 0.69 τ at large τ (355 steps at 512). Call it
        after the weight init, on a fresh network only."""
        with torch.no_grad():
            self.proj.bias[:self.dim].uniform_(-math.log(tau_max), 0.0)

    def step(self, x: torch.Tensor, h: torch.Tensor, done: torch.Tensor, inj: Optional[torch.Tensor] = None,
             h_inj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B, D], h [B, D] (previous state), done [B] (1 = x starts an episode) -> new h [B, D]. ``inj`` [B]
        (1 = inject) and ``h_inj`` [B, D]: the injected state (module docstring)."""
        k, ht = self.proj(x).chunk(2, dim=-1)
        z = torch.sigmoid(k)
        keep = (1.0 - done.to(x.dtype))[:, None]
        if inj is None:
            return (1.0 - z) * keep * h + z * ht
        return (1.0 - z) * (keep * h + inj.to(x.dtype)[:, None] * h_inj) + z * ht

    def scan(self, x: torch.Tensor, h0: torch.Tensor, done: torch.Tensor, inj: Optional[torch.Tensor] = None,
             h_inj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [T, B, D], h0 [B, D] (state before step 0), done [T, B] -> h [T, B, D] (parallel). ``inj`` [T, B],
        ``h_inj`` [T, B, D]: injected states (module docstring)."""
        k, ht = self.proj(x).chunk(2, dim=-1)
        log_z = -F.softplus(-k)
        if inj is None:
            log_a = -F.softplus(k) - DONE_LOG * done.to(x.dtype)[..., None]
            return scan_log(log_a, log_z, ht, h0)
        log_c = -F.softplus(k)
        log_a = log_c - DONE_LOG * done.to(x.dtype)[..., None]
        return scan_log(log_a, log_z, ht, h0, inj=inj, log_c=log_c, h_inj=h_inj)

    def scan_sequential(self, x: torch.Tensor, h0: torch.Tensor, done: torch.Tensor,
                        inj: Optional[torch.Tensor] = None, h_inj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reference: ``step`` in a Python loop (tests)."""
        out, h = [], h0
        for t in range(x.shape[0]):
            if inj is None:
                h = self.step(x[t], h, done[t])
            else:
                h = self.step(x[t], h, done[t], inj[t], h_inj[t])
            out.append(h)
        return torch.stack(out)
