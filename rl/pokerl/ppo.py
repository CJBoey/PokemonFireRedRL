"""Recurrent PPO: rollout storage on the GPU, collection, GAE, update.

* Rollout of T steps x N envs; the minGRU state is carried across rollouts and stored at the rollout start
  (``h0``); ``done[t]`` = observation t starts a life (CleanRL ``next_done`` convention).
* Rewards: raw reward divided by the running std of the discounted return (gamma, no mean subtraction,
  VecNormalize style), clipped to [-10, 10]. The policy also sees the previous action (one-hot; none at a life's
  start).
* GAE(gamma, lambda 0.95); every life end cuts the advantage recursion and the bootstrap (CleanRL: a truncation counts
  as a terminal state of value 0).
* Update: ``epochs`` x ``minibatches``, each N / minibatches whole env sequences (all T steps, parallel minGRU scan
  from the stored h0); clipped surrogate (eps 0.2), unclipped value MSE (coef 0.25), masked entropy (coef
  ``ent_coef``; the policy terms averaged over decision steps only), advantages normalised per minibatch, Adam
  (eps 1e-5), global grad-norm clip 0.5; the update stops (rest of the epoch and later epochs) once a minibatch's
  approx KL exceeds ``target_kl``. ``model.sequence`` is compiled (``torch.compile``, ``PPOConfig.compile``) and
  runs under bf16 autocast (``PPOConfig.autocast``); the losses, their reductions and Adam stay fp32.
* Collection (:class:`Collector`): one CUDA-graph replay per step and env group uploads the staged observation,
  runs the policy (``model.step`` compiled, ``PPOConfig.compile_step``) and writes the step into the rollout store;
  the collector's CUDA stream has the highest priority. The pipelined collector (several env groups) and the
  overlapped learner (:class:`Learner`, the update of rollout k on its own CUDA stream while rollout k+1 is collected
  by a separate acting copy of the model) hide the update behind the collection. The learner replays one CUDA graph
  per minibatch step (:class:`UpdateGraphs`, ``PPOConfig.graph_update``): ~3 ms of host time per update instead of
  ~75 ms of kernel launches on the collector's thread.
* Screen: the staging block, the step graphs' inputs and the rollout store hold the frames (``pixels``, uint8 [n,
  80, 120]: the envs' mapped frames, page-locked once with ``cudaHostRegister`` and uploaded by the step graph
  straight from the workers' shared memory, ``PPOConfig.pin_pixels``, ``Staging.pixels_host``; else copied into the
  pinned block) and the trail plane (``trail``, uint8 [n, 11, 15], written by the env straight into the staging
  block).
* Rewind (on when the envs' ``EnvConfig.rewind`` is): the collector keeps, per env and capture slot, the policy
  state of every capture (:class:`PolicyStateStore`: the recurrent state it carries into the captured observation
  and the action taken on the observation before). A rewound row (``info["rewind"]``: ``done`` 1, a GAE boundary)
  gets that state back before the next policy step: ``h`` overwritten, ``prev_action`` restaged, ``inj`` = 1 staged;
  the step computes the minGRU with the injection (mingru.py) and writes ``inj`` and ``inj_entry`` into the rollout
  store. The injected states themselves are in the store's :class:`InjectionPool`, ``inj_entry`` being a row's entry
  there (0, always zero: no injection). The learner gathers its minibatch's states from the pool and its scan injects
  them (``Policy.sequence(..., inj, h_inj)``). Off: nothing is allocated and every computation is the one without
  rewind.
"""
from __future__ import annotations

import collections
import contextlib
import gc
import threading
import warnings
from concurrent import futures
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .mask import NUM_ACTIONS
from .model import NO_ACTION, Policy, TRAIL_SHAPE, masked_entropy
from .reward import TERMS


@dataclass
class PPOConfig:
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.25
    ent_coef: float = 0.03
    lr: float = 2.5e-4
    lr_warmup: int = 100            # lr * k / lr_warmup in the k-th update of a fresh run and after a resume
                                    # (k < lr_warmup), then lr (0 = off; warmup_lr, Learner.set_lr)
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    epochs: int = 1
    minibatches: int = 64
    target_kl: float = 0.03
    reward_clip: float = 10.0
    compile: bool = True            # torch.compile of the update's loss (CUDA only)
    autocast: bool = True           # bf16 autocast around the update's forward pass (CUDA only)
    compile_step: bool = True       # torch.compile(model.step) inside the collector's step graphs (CUDA only)
    graph_update: bool = True       # the Learner replays one CUDA graph per minibatch step and for GAE (CUDA only)
    cudnn_benchmark: bool = True    # prepare_learner: torch.backends.cudnn.benchmark
    pin_pixels: bool = True         # the step graphs upload the screen pixels straight from the workers'
                                    # shared-memory frames (page-locked with cudaHostRegister) instead of the collector
                                    # thread copying every frame into the pinned staging block first


def warmup_lr(cfg: PPOConfig, k: int) -> float:
    """The learning rate of the k-th update of a fresh run or after a resume (k = 1, 2, ...): ``lr * k / lr_warmup``
    while ``k < lr_warmup`` (a linear ramp), else exactly ``lr`` (also with ``lr_warmup`` 0)."""
    w = int(cfg.lr_warmup)
    if w > 0 and 1 <= k < w:
        return cfg.lr * k / w
    return cfg.lr


class RewardScaler:
    """Divides rewards by the running std of the discounted return (per env return, shared stats).

    Per-term breakdown (logging only): when called with ``terms`` [n, K] (the reward split into K terms summing to
    it), each env also keeps a discounted return per term G_k (reset with G), and the scaler sums G_k and G_k * G over
    the same samples as the variance (all history; float64 running sums). :meth:`std_shares` gives Cov(G_k, G) /
    Var(G): the terms' shares of the variance the scaler divides by (sum 1; a term that moves with the others counts
    its covariance with them, a negative share pulls the variance down)."""

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, eps: float = 1e-8):
        self.gamma, self.clip, self.eps = gamma, clip, eps
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4
        self.ret_terms = None  # [num_envs, K] per-term discounted returns (allocated on the first terms call)
        self.t_sum = self.t_cross = None  # [K] sums of G_k and of G_k * G over the term samples
        self.t_count = 0.0

    def _update(self, x: np.ndarray) -> None:
        bm, bv, bc = float(x.mean()), float(x.var()), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def _update_terms(self, xk: np.ndarray) -> None:
        """Adds a batch of per-term returns (xk: [n, K]; G = the row sum) to the running sums."""
        x = xk @ np.ones(xk.shape[1])  # matrix-vector products: faster than axis sums
        self.t_sum += np.ones(xk.shape[0]) @ xk
        self.t_cross += x @ xk
        self.t_count += xk.shape[0]

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + self.eps))

    def std_shares(self):
        """[K] Cov(G_k, G) / Var(G) over the term samples (None before any terms call)."""
        if self.t_sum is None or self.t_count == 0:
            return None
        mk = self.t_sum / self.t_count
        cov = self.t_cross / self.t_count - mk * mk.sum()
        v = float(cov.sum())
        return cov / v if v > 0 else np.zeros_like(cov)

    def __call__(self, reward: np.ndarray, done: np.ndarray, ids=slice(None), terms=None) -> np.ndarray:
        """Scales the rewards of envs ``ids`` (default all; a slice for one env group of the pipelined
        collector: the statistics are shared, updated with each group's batch in turn). ``terms`` [n, K]:
        the reward's terms, for the per-term breakdown (optional)."""
        ret = self.ret[ids] * self.gamma + reward
        self._update(ret)
        out = np.minimum(np.maximum(reward / self.std, -self.clip), self.clip).astype(np.float32)  # (= np.clip, cheaper)
        ret[done] = 0.0
        self.ret[ids] = ret
        if terms is not None:
            if self.ret_terms is None or self.ret_terms.shape[1] != terms.shape[1]:
                self.ret_terms = np.zeros((len(self.ret), terms.shape[1]), dtype=np.float64)
                self.t_sum, self.t_cross = np.zeros(terms.shape[1]), np.zeros(terms.shape[1])
                self.t_count = 0.0
            rt = self.ret_terms[ids]  # a view for a slice (the collector's groups)
            rt *= self.gamma
            rt += terms
            self._update_terms(rt)
            rt[done] = 0.0
            self.ret_terms[ids] = rt
        return out

    def state_dict(self):
        d = {"ret": self.ret.copy(), "mean": self.mean, "var": self.var, "count": self.count}
        if self.ret_terms is not None:
            d.update(ret_terms=self.ret_terms.copy(), t_sum=self.t_sum.copy(), t_cross=self.t_cross.copy(),
                     t_count=self.t_count)
        return d

    def load_state_dict(self, d):
        if len(d["ret"]) == len(self.ret):
            self.ret[:] = d["ret"]
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
        if "ret_terms" in d and len(d["ret_terms"]) == len(self.ret):
            self.ret_terms = np.array(d["ret_terms"], dtype=np.float64)
            self.t_sum, self.t_cross = np.array(d["t_sum"], dtype=np.float64), np.array(d["t_cross"], dtype=np.float64)
            self.t_count = float(d["t_count"])


_TDT = {np.dtype(np.int64): torch.int64, np.dtype(np.float32): torch.float32, np.dtype(np.uint8): torch.uint8,
        np.dtype(bool): torch.bool}
_NDT = {v: k for k, v in _TDT.items()}

# Per-step fields the policy step reads (staged on the host, uploaded once per step) and stores in the rollout.
OBS_KEYS = ("pixels", "trail", "prev_action")


def store_fields(pixels: Tuple[int, ...]) -> tuple:
    """(rollout attribute, shape after [rows, N], dtype) of every rollout field without the rewind's."""
    return (("pixels", tuple(pixels), torch.uint8), ("trail", TRAIL_SHAPE, torch.uint8),
            ("prev_action", (), torch.int64), ("mask", (NUM_ACTIONS,), torch.bool), ("done", (), torch.float32),
            ("actions", (), torch.int64), ("logprobs", (), torch.float32), ("values", (), torch.float32))


def rewind_fields(layers: int, hidden: int):
    """The fields of a standalone rollout for the rewind: ``inj`` (1 = the policy state was injected at this step)
    and ``h_inj`` (the injected state, zero where no injection)."""
    return (("inj", (), torch.float32), ("h_inj", (layers, hidden), torch.float32))


def store_rewind_fields():
    """The same fields of a :class:`RolloutStore`: ``inj`` and ``inj_entry``, the state's entry in the store's
    :class:`InjectionPool` (0 where no injection)."""
    return (("inj", (), torch.float32), ("inj_entry", (), torch.int64))


class Rollout:
    """GPU storage for T steps of N envs. Standalone (own tensors) or a [T, N] window (rows
    ``index*T .. index*T+T-1``) of a :class:`RolloutStore` that the collector's step graphs write into."""

    def __init__(self, T: int, N: int, layers: int, hidden: int, device, store: Optional["RolloutStore"] = None,
                 index: int = 0, rewind: bool = False, pixels: Tuple[int, ...] = (80, 120)):
        """``pixels``: the frame shape (standalone rollouts; a store's window has the store's fields)."""
        self.T, self.N = T, N
        z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=device)  # noqa: E731
        if store is None:
            fields = store_fields(pixels) + (rewind_fields(layers, hidden) if rewind else ())
            f = {k: z(T, N, *shape, dtype=dt) for k, shape, dt in fields}
        else:
            assert store.T == T and store.N == N
            f = {k: v[index * T:(index + 1) * T] for k, v in store.fields.items()}
        self.index = index
        self.obs = {k: f[k] for k in OBS_KEYS}
        self.mask, self.done = f["mask"], f["done"]
        self.actions, self.logprobs, self.values = f["actions"], f["logprobs"], f["values"]
        self.inj, self._h_inj = f.get("inj"), f.get("h_inj")  # rewind (None when off); dense when standalone
        self.inj_entry = f.get("inj_entry")  # (a store's rollout: the entries in the store's pool instead)
        self.inj_pool = store.inj_pool.h if self.inj_entry is not None else None
        self.rewards = z(T, N)
        self.advantages = z(T, N)
        self.returns = z(T, N)
        self.h0 = z(layers, N, hidden)
        # bootstrap inputs of GAE (value and episode-start flag of the observation after the last step: the scratch
        # row's)
        self.next_value = z(N)
        self.next_done = z(N)

    @property
    def h_inj(self) -> Optional[torch.Tensor]:
        """The injected states [T, N, layers, H], zero where no injection (None when off): the field of a standalone
        rollout, a new dense tensor from the pool for a store's (not for the learner, which gathers its minibatches)."""
        return self._h_inj if self.inj_entry is None else self.inj_pool[self.inj_entry]


class InjectionPool:
    """The injected policy states of a :class:`RolloutStore`: ``h`` [capacity + 1, layers, H], a ring of ``capacity``
    entries after entry 0, which stays zero (a rollout row without an injection, ``inj_entry`` 0). The collector keeps
    a state here when it overwrites a row's carried state and stages the entry; the step graphs store the staged
    entry in the row like any other input, including the bootstrap step and the next rollout's first step, which act
    on the same staged observation.

    The ring must not overwrite what the learner still reads: the buffer being collected and the previous one, whose
    row 0 may have been staged at the very end of the collection before it. So :meth:`keep` gives entry 0 instead when
    the entries since the start of that collection fill the ring: the policy acted on the injected state, but the
    learner sees an injection of zeros at that row. :meth:`begin_rollout` warns if that happened."""

    def __init__(self, capacity: int, layers: int, hidden: int, device):
        self.capacity = int(capacity)
        self.h = torch.zeros(self.capacity + 1, layers, hidden, device=device)
        self.next = 0  # entries handed out so far (entry = 1 + next % capacity)
        self.starts = collections.deque([0], maxlen=3)  # ``next`` at the start of this collection and the two before
        self.dropped = 0

    def begin_rollout(self) -> None:
        """At the start of every collection."""
        self.starts.append(self.next)
        if self.dropped:
            print(f"WARNING injection pool full: {self.dropped} injected rows reached the learner as zeros "
                  f"({self.capacity} entries)", flush=True)
            self.dropped = 0

    def keep(self, h: torch.Tensor, rows: np.ndarray) -> np.ndarray:
        """Copies ``h[:, rows]`` (a group's state) into new entries on the current stream, one copy kernel per row
        (no index upload, few rows); returns the entries (int64, 0 for a row that found the ring full)."""
        entries = np.zeros(rows.size, dtype=np.int64)
        for j, r in enumerate(rows.tolist()):
            if self.next - self.starts[0] >= self.capacity:
                self.dropped += rows.size - j
                break
            entries[j] = e = 1 + self.next % self.capacity
            self.next += 1
            self.h[e].copy_(h[:, r])
        return entries


class RolloutStore:
    """``buffers`` rollouts of [T, N] in one tensor per field, plus one scratch row (index ``buffers*T``)
    where the bootstrap step writes. The step graphs write row ``slot`` (a device scalar in the staging
    block), so one captured graph serves every step of every buffer."""

    def __init__(self, buffers: int, T: int, N: int, device, pixels: Tuple[int, ...],
                 rewind: Optional[Tuple[int, int]] = None):
        """``pixels``: the frame shape; ``rewind``: (layers, hidden) to add the rewind fields ``inj`` / ``inj_entry``
        and ``inj_pool``."""
        self.buffers, self.T, self.N = buffers, T, N
        self.rows = buffers * T + 1
        self.scratch = buffers * T
        fields = store_fields(pixels) + (store_rewind_fields() if rewind is not None else ())
        self.fields = {k: torch.zeros(self.rows, N, *shape, dtype=dt, device=device) for k, shape, dt in fields}
        # (at 1024 envs 16,384 entries, 64 MiB)
        self.inj_pool = InjectionPool(max(4096, 16 * N), *rewind, device) if rewind is not None else None

    @property
    def nbytes(self) -> int:
        """Bytes of the store's tensors."""
        pool = 0 if self.inj_pool is None else self.inj_pool.h.numel() * self.inj_pool.h.element_size()
        return pool + sum(v.numel() * v.element_size() for v in self.fields.values())


class Staging:
    """The inputs of one policy step for ``n`` envs in one byte block, field-major (each field a contiguous
    [n, ...] array: the env writes its trail plane in place). ``host`` (pinned when on CUDA) is written by
    the CPU, ``dev`` is what the step graph reads; one copy per step moves the whole block. Fields:
    ``prev_action``, ``slot`` (int64 [1]: the rollout row to store into), ``done``, ``mask``; with ``rewind`` also
    ``inj`` (float [n], 1 = inject the carried state, which the collector has overwritten), ``inj_entry`` (int64
    [n]: its :class:`InjectionPool` entry, stored only) and ``rw_save`` (int64 [n]: per row the entry of the group's
    policy-state store view that the step writes the carried state into, a trash entry unless the row was captured;
    :class:`PolicyStateStore`); then ``pixels`` (uint8 [n, *pixels]) and ``trail`` (uint8 [n, 11, 15])."""

    def __init__(self, n: int, device, pixels: Tuple[int, ...], rewind: bool = False,
                 pixels_host: Optional[np.ndarray] = None):
        """``pixels_host`` (CUDA): a C-contiguous uint8 [n, *pixels] array the frames are read from in place (the vec
        env's mapped frames): it is page-locked (``cudaHostRegister``, once per mapping) and ``upload`` copies it to
        the device field directly, so ``np["pixels"]`` is that array and nothing copies it on the host. The caller
        must not let the workers write it between a launch and the step's completion (the collector posts the actions
        only after the step's event: ``_Group.wait``). Not contiguous, not CUDA or the page-locking refused (a
        warning): the pinned block holds the frames and the caller copies."""
        spec = [("prev_action", (n,), torch.int64), ("slot", (1,), torch.int64), ("done", (n,), torch.float32),
                ("mask", (n, NUM_ACTIONS), torch.bool)]
        if rewind:
            spec += [("inj", (n,), torch.float32), ("inj_entry", (n,), torch.int64), ("rw_save", (n,), torch.int64)]
        spec += [("pixels", (n,) + tuple(pixels), torch.uint8), ("trail", (n,) + TRAIL_SHAPE, torch.uint8)]
        device = torch.device(device)
        cuda = device.type == "cuda"
        self.pixels_host = None
        if pixels_host is not None and cuda:
            want = (n,) + tuple(pixels)
            if not (isinstance(pixels_host, np.ndarray) and pixels_host.dtype == np.uint8
                    and tuple(pixels_host.shape) == want and pixels_host.flags.c_contiguous):
                warnings.warn(f"Staging: pixels_host {getattr(pixels_host, 'shape', None)} is not a C-contiguous uint8 "
                              f"{want} array: the frames are copied on the host")
            elif _host_register(pixels_host):
                self.pixels_host = pixels_host
        off, hoff, layout = 0, 0, []  # (device offset, host offset: the host block skips the in-place pixels)
        for k, shape, dt in spec:
            size = int(np.prod(shape)) * torch.tensor([], dtype=dt).element_size()
            layout.append((k, off, hoff, size, shape, dt))
            off += (size + 7) // 8 * 8
            if not (k == "pixels" and self.pixels_host is not None):
                hoff += (size + 7) // 8 * 8
        self.nbytes = off
        self.host = torch.zeros(hoff, dtype=torch.uint8, pin_memory=cuda)
        self.dev = torch.zeros(off, dtype=torch.uint8, device=device)
        hb = self.host.numpy()
        self.np = {k: hb[ho:ho + sz].view(_NDT[dt]).reshape(shape) for k, o, ho, sz, shape, dt in layout
                   if not (k == "pixels" and self.pixels_host is not None)}
        self.t = {k: self.dev[o:o + sz].view(dt).view(shape) for k, o, ho, sz, shape, dt in layout}
        self._copies = [(self.dev, self.host)]
        if self.pixels_host is not None:
            self.np["pixels"] = self.pixels_host
            (_, o, ho, sz, _, _), = [f for f in layout if f[0] == "pixels"]
            pad = (sz + 7) // 8 * 8
            src = torch.from_numpy(self.pixels_host.reshape(-1))
            self._copies = [(self.dev[:o], self.host[:ho]), (self.dev[o:o + sz], src),
                            (self.dev[o + pad:], self.host[ho:])]

    def upload(self) -> None:
        for dst, src in self._copies:
            dst.copy_(src, non_blocking=True)


_registered: list = []  # (keeps a reference to every page-locked array: the mapping must outlive the registration)


def _host_register(a: np.ndarray) -> bool:
    """Page-locks the pages of ``a`` for CUDA (``cudaHostRegister`` of the page-aligned range around it; a mapping of
    shared memory is fine). False with a warning when CUDA refuses (the caller copies instead)."""
    import mmap as _mmap
    ps = _mmap.PAGESIZE
    ptr, n = a.ctypes.data, a.nbytes
    start = ptr & ~(ps - 1)
    end = (ptr + n + ps - 1) & ~(ps - 1)
    try:
        torch.cuda.cudart().cudaHostRegister(start, end - start, 0)
        ok = torch.from_numpy(a.reshape(-1)).is_pinned()
    except Exception as e:  # noqa: BLE001
        ok = False
        warnings.warn(f"cudaHostRegister of {n} bytes failed ({type(e).__name__}: {e}): the frames are copied on the host")
    if ok:
        _registered.append(a)
    else:
        warnings.warn(f"cudaHostRegister of {n} bytes: the range is not pinned afterwards; the frames are copied on the host")
    return ok


class StepGraph:
    """The rollout step (forward pass, masked Gumbel-max sampling, state update) captured as one CUDA
    graph: a 256-env step is ~100 small kernels, launch-bound in eager mode (~1.4 ms vs ~0.2 ms).
    Inputs are the device views ``self.inp``, ``self.mask``, ``self.done`` of a :class:`Staging` block;
    ``self.h`` is the recurrent state (updated in place by each replay); outputs ``action``, ``logprob``,
    ``value`` are static tensors. With ``upload=True`` the graph starts with the host->device copy of the
    staging block; with a ``store`` it ends by writing the step's inputs and outputs into row
    ``staging.t["slot"]`` of the store, columns ``cols``, and copying the actions to ``host_action``
    (pinned): one replay per step and no other launch. The parameters are read in place, so parameter updates
    (optimiser steps, or the copy into the acting model) are seen by later replays. Do not call
    ``torch.cuda.empty_cache()`` while a StepGraph is alive: it frees the cuBLAS workspace the captured kernels use
    (illegal memory access on a later replay). With an ``inj`` field in the staging block (rewind) the step injects
    the carried state where ``inj`` = 1 (``step_fn(..., inj, h)``) and stores ``inj`` and ``inj_entry``; with
    ``rw_dst`` (the group's view of the policy-state store, [layers, n * (slots + 1), H]) it first writes the carried
    state of row i into entry ``rw_save[i]`` (the captures' policy state, without a host round trip; the other rows
    into their trash entries).

    ``capture_stream``: the stream the graph is warmed up and captured on (the collector passes its own stream).
    A captured graph keeps the cuBLAS workspace of its capture stream (PyTorch keys it by handle and stream), so two
    graphs captured on one CUDA stream share a workspace and corrupt each other's matrix products when they replay
    concurrently on different streams. None: a new side stream for the warm-up and ``torch.cuda.graph``'s
    class-wide default capture stream, i.e. the same stream for every such graph of the process: only for graphs
    that never replay concurrently with others."""

    def __init__(self, model: Policy, n: int, device, use_graph: bool = True, staging: Optional[Staging] = None,
                 store: Optional[RolloutStore] = None, cols: Optional[slice] = None, upload: bool = False,
                 step_fn=None, rw_dst: Optional[torch.Tensor] = None,
                 capture_stream: Optional[torch.cuda.Stream] = None):
        self.model, self.n = model, n
        self.capture_stream = capture_stream
        self.step_fn = step_fn if step_fn is not None else model.step  # e.g. torch.compile(model.step)
        self.staging = st = staging if staging is not None else Staging(n, device, model.pixel_shape)
        self.upload = upload
        self.inp = {k: st.t[k] for k in OBS_KEYS}
        self.mask, self.done = st.t["mask"], st.t["done"]
        self.inj, self.inj_entry = st.t.get("inj"), st.t.get("inj_entry")  # rewind (None when off)
        self.rw_dst, self.rw_save = rw_dst, st.t.get("rw_save")
        self.mask.fill_(True)
        self.h = model.initial_state(n, device)
        self.store = store
        if store is not None:
            cols = cols if cols is not None else slice(0, n)
            self._dst = {k: v[:, cols] for k, v in store.fields.items()}
        cuda = torch.device(device).type == "cuda"
        self.host_action = torch.zeros(n, dtype=torch.int64, pin_memory=cuda)
        self.graph = None
        if use_graph and cuda:
            st.np["slot"][0] = store.scratch if store is not None else 0
            h_save = self.h.clone()
            s = capture_stream if capture_stream is not None else torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    self._body()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=capture_stream), torch.no_grad():
                self._body()
            self.h.copy_(h_save)

    def _body(self):
        st = self.staging
        if self.upload:
            st.upload()
        if self.rw_dst is not None:  # rewind: the carried state of this step's captures into the policy-state store
            self.rw_dst.index_copy_(1, self.rw_save, self.h)
        if self.inj is None:
            logits, value, h = self.step_fn(self.inp, self.h, self.done, self.mask)
        else:
            logits, value, h = self.step_fn(self.inp, self.h, self.done, self.mask, self.inj, self.h)
        logp = torch.log_softmax(logits, dim=-1)
        gumbel = -torch.log(-torch.log(torch.rand_like(logp).clamp_(1e-20, 1.0)))
        action = torch.argmax(logp + gumbel, dim=-1)
        self.action, self.value = action, value
        self.logprob = logp.gather(-1, action[:, None]).squeeze(-1)
        self.h.copy_(h)
        if self.store is not None:
            slot, d = st.t["slot"], self._dst
            src = dict(self.inp, mask=self.mask, done=self.done, actions=action, logprobs=self.logprob, values=value)
            if self.inj is not None:
                src["inj"], src["inj_entry"] = self.inj, self.inj_entry
            for k, v in src.items():
                d[k].index_copy_(0, slot, v[None])
        if self.upload:
            self.host_action.copy_(action, non_blocking=True)

    @torch.no_grad()
    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        return self.action, self.logprob, self.value


class _Group:
    """One PokeRLEnv of the collector, its columns of the rollout, staging block and step graph."""

    def __init__(self, env, cols: slice, model: Policy, store: RolloutStore, device, use_graph: bool, step_fn=None,
                 rw_store: Optional["PolicyStateStore"] = None, capture_stream: Optional[torch.cuda.Stream] = None,
                 pin_pixels: bool = True):
        self.env, self.cols = env, cols
        n = self.n = env.num_envs
        self.staging = Staging(n, device, model.pixel_shape, rewind=rw_store is not None,
                               pixels_host=getattr(env, "_pix", None) if pin_pixels else None)
        self.np = self.staging.np
        po = getattr(env, "plane_out", None) or {}
        if "trail" in po:  # the env writes the trail plane straight into the pinned staging block
            po["trail"] = self.np["trail"]
        rw_dst = self.rw_trash = None
        if rw_store is not None:  # (rewind) every row's store write goes to its trash entry unless captured
            rw_dst = rw_store.group_view(cols)
            self.rw_trash = np.arange(n, dtype=np.int64) * rw_store.stride + rw_store.slots
            np.copyto(self.np["rw_save"], self.rw_trash)
        self.step = StepGraph(model, n, device, use_graph=use_graph, staging=self.staging, store=store, cols=cols,
                              upload=True, step_fn=step_fn, rw_dst=rw_dst, capture_stream=capture_stream)
        self.event = torch.cuda.Event() if torch.device(device).type == "cuda" else None
        self.h_save = torch.zeros_like(self.step.h)
        self.action = self.step.host_action.numpy()
        self.t = -1  # step whose results are pending (posted, not yet collected)
        self.begun = False  # step_wait_begin called, step_wait_end pending

    def stage(self, enc, mask, done, action) -> None:
        """Writes the next observation (after ``action``, ``done`` bool [n]) into the host block (``enc``: the env's
        ``trail`` plane, the block's own when written in place, and its mapped ``pixels``: the block's own with
        ``pixels_host`` (uploaded in place), else copied here)."""
        hn = self.np
        for k, v in enc.items():
            if v is not hn[k]:
                np.copyto(hn[k], v)
        np.copyto(hn["mask"], mask)
        np.copyto(hn["done"], done)  # (bool -> 0/1 float)
        pa = hn["prev_action"]
        if action is None:
            pa.fill(NO_ACTION)
        else:
            np.copyto(pa, action)
            if done.any():
                pa[done] = NO_ACTION
        if "inj" in hn:
            hn["inj"].fill(0.0)  # (rewound rows: set by Collector._rewind_host)
            hn["inj_entry"].fill(0)
            np.copyto(hn["rw_save"], self.rw_trash)  # (captured rows: set by Collector._rewind_host)

    def act(self, slot: int, stream) -> np.ndarray:
        """Policy step on the staged observation, stored in rollout row ``slot``; returns the actions (host)."""
        self.launch(slot, stream)
        return self.wait()

    def launch(self, slot: int, stream) -> None:
        """Starts ``act`` without waiting for it: the staging block must not be written before ``wait``."""
        self.np["slot"][0] = slot
        if self.event is None:
            self.step()
            return
        with torch.cuda.stream(stream):
            self.step()
            self.event.record()

    def ready(self) -> bool:
        """The launched policy step has finished (``wait`` returns at once)."""
        return self.event is None or self.event.query()

    def wait(self) -> np.ndarray:
        """The actions of the launched policy step (host), once it has finished."""
        if self.event is not None:
            self.event.synchronize()
        return self.action


class PolicyStateStore:
    """The policy state of every live capture, ``[N, slots]`` entries keyed by (env, capture slot; the env's ring
    hands out the slots, rewind.py): ``h`` [layers, N, slots, H] (fp32, on the device), ``prev_action`` (host) and the
    capture's ``serial`` (-1: none), which every load checks.

    What is stored for a capture of env i (``info["capture"]`` of the step whose returned observation o is the
    captured state): the recurrent state the collector carries into o (the one its policy step on the observation
    before o produced: the step graph's ``h`` when the step's results are collected) and the action taken on that
    observation (o's staged ``prev_action``).

    No host round trip and no synchronisation: ``h`` is a view of ``[layers, N, slots + 1, H]`` whose extra entry per
    env is a trash entry, so that the collector's step graph writes the carried state of every row each step
    (``index_copy_`` with the staged indices ``rw_save``, :meth:`group_view`): captured rows into their entry, the
    others into their trash entry; the host side of a capture is :meth:`note`. :meth:`load` (a rewind, before the
    policy step) and :meth:`save` (the eager equivalent of the step's write) copy rows device to device on the current
    stream: up to ``LOOP_MAX`` rows one copy kernel per row (Python ints index the views: no index upload), more rows
    one ``index_select`` + ``index_copy_`` with the indices uploaded from pinned memory without blocking.
    ``prev_action`` / ``serial`` stay on the host, next to the staging block they are restaged into."""

    LOOP_MAX = 8  # rows per save / load copied one by one; more: one indexed gather + scatter

    def __init__(self, num_envs: int, slots: int, layers: int, hidden: int, device):
        self.device = torch.device(device)
        self.slots = int(slots)
        self.stride = self.slots + 1  # entries per env: the slots and a trash entry
        self._full = torch.zeros(layers, num_envs, self.stride, hidden, device=self.device)
        self.h = self._full[:, :, :self.slots]
        self._h_flat = self._full.view(layers, num_envs * self.stride, hidden)  # entry (c, s): column c * stride + s
        self.prev_action = np.full((num_envs, slots), NO_ACTION, dtype=np.int64)
        self.serial = np.full((num_envs, slots), -1, dtype=np.int64)

    def group_view(self, cols: slice) -> torch.Tensor:
        """The entries of collector columns ``cols`` as [layers, n * stride, H] (a view): local row i's entry s is
        column ``i * stride + s``, ``s = slots`` its trash entry."""
        layers, _, _, hidden = self._full.shape
        return self._full[:, cols].view(layers, -1, hidden)

    def note(self, cols, slots, prev_action, serials) -> None:
        """The host side of storing captures (cols = the collector's env indices): previous action and serial; their
        ``h`` is written by the step graph (``rw_save``) or :meth:`save`."""
        self.prev_action[cols, slots] = prev_action
        self.serial[cols, slots] = serials

    def _upload(self, a: np.ndarray, b: np.ndarray):
        """Two int64 index vectors of one length on the device, in one non-blocking copy from pinned memory (the
        caching host allocator keeps the pinned block until the copy has run)."""
        x = torch.from_numpy(np.concatenate([a, b]).astype(np.int64, copy=False))
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        return x[:a.size], x[a.size:]

    def save(self, h: torch.Tensor, rows, cols, slots, prev_action, serials) -> None:
        """Stores ``h[:, rows]`` (a group's state, rows = its env indices) as the entries (cols, slots) (cols = the
        collector's env indices); on the current stream."""
        rows, cols, slots = (np.asarray(x, dtype=np.int64) for x in (rows, cols, slots))
        self.note(cols, slots, prev_action, serials)
        flat = cols * self.stride + slots
        if rows.size <= self.LOOP_MAX:
            for r, f in zip(rows.tolist(), flat.tolist()):
                self._h_flat[:, f].copy_(h[:, r])
        else:
            r, f = self._upload(rows, flat)
            self._h_flat.index_copy_(1, f, h.index_select(1, r))

    def load(self, h: torch.Tensor, rows, cols, slots, serials):
        """Writes the entries (cols, slots) into ``h[:, rows]`` (current stream); returns their prev_action. Raises
        if an entry does not hold the policy state of capture ``serials``."""
        rows, cols, slots = (np.asarray(x, dtype=np.int64) for x in (rows, cols, slots))
        bad = self.serial[cols, slots] != np.asarray(serials)
        if bad.any():
            j = int(np.flatnonzero(bad)[0])
            raise RuntimeError(f"rewind of env {int(cols[j])} to capture {int(np.asarray(serials)[j])}: "
                               f"its policy state was not stored (entry holds {int(self.serial[cols, slots][j])})")
        flat = cols * self.stride + slots
        if rows.size <= self.LOOP_MAX:
            for r, f in zip(rows.tolist(), flat.tolist()):
                h[:, r].copy_(self._h_flat[:, f])
        else:
            r, f = self._upload(rows, flat)
            h.index_copy_(1, r, self._h_flat.index_select(1, f))
        return self.prev_action[cols, slots]


# CUDA stream priority of the collector: the highest (lower = higher; the learner stream is 0, the lowest)
COLLECTOR_PRIORITY: Optional[int] = None  # None = torch.cuda.Stream.priority_range()[1]


class Collector:
    """Runs the acting policy in one or more PokeRLEnv groups and fills the rollouts of a RolloutStore;
    keeps the env/recurrent state between calls.

    * One env (``collect`` with one group) is the synchronous loop: act, post, wait + host work, repeat.
    * Several envs (the pipelined collector): group g's columns of the rollout are ``g * N/G ..``; each step, for each
      group in turn: wait for its workers and do its host work (reward, life ends, observation, mask), run its policy
      step, post its next actions. While one group's host work and policy step run, the other groups' workers
      simulate. Each group has its own step graph (captured at n = N/G), recurrent state slice, done/prev_action; the
      reward scaler is shared (one running std of the discounted return for all N envs, as with one env: the value
      head sees one return scale in every column).
    * ``model`` is the acting policy (its parameters are read in place by the graphs); ``rollouts`` are
      ``buffers`` [T, N] views of the store, filled by ``collect(ro)``.
    * ``async_reset`` (default with several groups): a group's life resets run in a helper thread
      (PokeRLEnv.step_wait_begin) while the other groups keep stepping; groups may then be a few steps
      apart within a rollout (each still writes its own columns, row = its own step index).
    * ``post_ahead`` (with ``async_reset``): a group's policy step is launched and the collector goes on with the
      next groups' host work while the GPU runs it; the actions are posted at the first checkpoint (before / after
      a group's ``step_wait_begin``, after its host work) at which the step has finished, and waited for at the
      latest at the ``G``-th checkpoint after the launch (G = number of groups), or before the group's own
      ``step_wait_begin``. The rollouts are the same as without it: each group's steps, inputs and outputs are
      unchanged, only the order of the CPU work across groups differs.
    * GPU work runs on ``self.stream`` (high priority, so the policy step is not queued behind a learner's
      kernels); ``collect`` orders itself after the caller's current stream at the start and the caller's
      current stream after it at the end.
    * Rewind (the envs' ``cfg.rewind``): ``rw_store`` (:class:`PolicyStateStore`) and the rollout fields
      ``inj``/``inj_entry``; see the module docstring.
    """

    def __init__(self, envs, model: Policy, cfg: PPOConfig, device, rollout: int = 128, buffers: int = 1,
                 seed: int = 0, cuda_graph: bool = True, async_reset: Optional[bool] = None,
                 post_ahead: bool = True):
        envs = list(envs) if isinstance(envs, (list, tuple)) else [envs]
        self.envs, self.model, self.cfg, self.device = envs, model, cfg, torch.device(device)
        self.N = n = sum(e.num_envs for e in envs)
        self.T = rollout
        self.rewind = any(bool(e.cfg.rewind) for e in envs)
        self.rw_store = None
        if self.rewind:
            slots = max(int(e.cfg.rewind_ring) for e in envs)
            self.rw_store = PolicyStateStore(n, slots, model.layers, model.hidden, device)
        self.pixels = tuple(model.pixel_shape)
        for e in envs:
            if tuple(e.pixel_shape) != self.pixels:
                raise ValueError(f"Collector: the env's pixels {e.pixel_shape} differ from the policy's {self.pixels}")
        self.store = RolloutStore(buffers, rollout, n, device, self.pixels,
                                  rewind=(model.layers, model.hidden) if self.rewind else None)
        self.rollouts = [Rollout(rollout, n, model.layers, model.hidden, device, store=self.store, index=k)
                         for k in range(buffers)]
        self.scaler = RewardScaler(n, cfg.gamma, cfg.reward_clip)
        cuda = self.device.type == "cuda"
        prio = COLLECTOR_PRIORITY if COLLECTOR_PRIORITY is not None else (
            torch.cuda.Stream.priority_range()[1] if cuda else 0)
        self.stream = torch.cuda.Stream(device=self.device, priority=prio) if cuda else None
        self.groups: List[_Group] = []
        # the policy step compiled (fewer, fused kernels in the captured graph); its fp32 numerics differ from eager
        # only by kernel fusion
        step_fn = torch.compile(model.step, dynamic=False) if (cuda and cuda_graph and cfg.compile_step) else None
        c = 0
        for e in envs:
            self.groups.append(_Group(e, slice(c, c + e.num_envs), model, self.store, device, cuda_graph, step_fn,
                                      rw_store=self.rw_store, capture_stream=self.stream, pin_pixels=cfg.pin_pixels))
            c += e.num_envs
        self._rew_host = [torch.zeros(rollout, n, pin_memory=cuda) for _ in range(buffers)]
        self._rew_event = [torch.cuda.Event() if cuda else None for _ in range(buffers)]
        self._poll = None
        self.posted_steps = 0
        # asynchronous resets (default: with several groups)
        self.async_reset = len(self.groups) > 1 if async_reset is None else bool(async_reset)
        self.post_ahead = bool(post_ahead) and self.async_reset
        for g in self.groups:
            g.deadline = 0  # post_ahead: checkpoints left before the launched step is waited for
        for g in self.groups:
            enc, mask = g.env.reset()
            g.stage(enc, mask, np.ones(g.n, dtype=bool), None)

    @property
    def h(self) -> torch.Tensor:
        return torch.cat([g.step.h for g in self.groups], dim=1)

    def _host(self, g: _Group, rew: np.ndarray, acc: Dict) -> None:
        """Collects group g's pending step: env host work, scaled reward into row g.t, stages the next obs.
        (After ``g.env.step_wait_begin`` when ``g.begun``.)"""
        a_np = g.action
        if g.begun:
            g.begun = False
            enc, mask, r, term, trunc, info = g.env.step_wait_end()
        else:
            enc, mask, r, term, trunc, info = g.env.step_wait()
        t0 = time.perf_counter()
        done = term | trunc
        terms = info["terms_stack"]  # (the env's [n, K] stack of its terms; the float64 sums upcast it)
        rs = self.scaler(r, done, g.cols, terms)
        rew[g.t, g.cols] = rs
        acc["raw"] += float(r.sum())
        acc["terms"] += np.ones(len(terms)) @ terms
        acc["waiting"] += int(info["waiting"].sum())
        acc["act_hist"] += np.bincount(a_np, minlength=NUM_ACTIONS)
        off = g.cols.start  # episode dicts carry the env index of the whole collector
        acc["episodes"].extend(dict(e, env=e["env"] + off) if off else e for e in info["episodes"])
        g.stage(enc, mask, done, a_np)
        if self.rw_store is not None:
            self._rewind_host(g, info)
        acc["t_host"] += time.perf_counter() - t0

    def _rewind_host(self, g: _Group, info: Dict) -> None:
        """After staging group g's next observation: stores the policy state of this step's captures (the state
        the step graph carries into the captured observations, their staged prev action: host side here, the state
        itself by the next replay through the staged ``rw_save``) and restores it for this step's rewinds (``h``
        overwritten on the collector's stream, prev action restaged, ``inj`` = 1), before the next policy step.
        Nothing on a step without either."""
        cap, rw = info.get("capture"), info.get("rewind")
        save = cap is not None and len(cap["rows"]) > 0
        load = rw is not None and len(rw["rows"]) > 0
        if not (save or load):  # (most steps)
            return
        st, hn, off = self.rw_store, g.np, g.cols.start
        if save:  # host side now; the step graph's next replay copies the carried state (staged rw_save)
            rows = np.asarray(cap["rows"], dtype=np.int64)
            slots = np.asarray(cap["slots"], dtype=np.int64)
            st.note(rows + off, slots, hn["prev_action"][rows], cap["serials"])
            hn["rw_save"][rows] = rows * st.stride + slots
        if load:
            rows = np.asarray(rw["rows"], dtype=np.int64)
            with torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext():
                pa = st.load(g.step.h, rows, rows + off, rw["slots"], rw["serials"])
                hn["inj_entry"][rows] = self.store.inj_pool.keep(g.step.h, rows)
            hn["prev_action"][rows] = pa
            hn["inj"][rows] = 1.0

    def _act_post(self, g: _Group, slot: int, t: int, acc: Dict) -> None:
        self._launch(g, slot, t, acc)
        self._post(g, acc)

    def _launch(self, g: _Group, slot: int, t: int, acc: Dict) -> None:
        """Starts group g's policy step for its step t (rollout row ``slot``); ``_post`` sends the actions."""
        t0 = time.perf_counter()
        g.launch(slot, self.stream)
        g.t = t
        g.deadline = len(self.groups)
        # (counted at the launch: the step acts with the actor's parameters of now, the caller's policy lag)
        self.posted_steps += g.n
        acc["t_act"] += time.perf_counter() - t0

    def _post(self, g: _Group, acc: Dict) -> None:
        """Waits for group g's launched policy step and posts its actions to the workers."""
        t0 = time.perf_counter()
        a = g.wait()
        t1 = time.perf_counter()
        g.env.step_async(a)
        acc["t_act"] += t1 - t0
        acc["t_act_wait"] += t1 - t0
        if self._poll is not None:  # while the workers simulate: a piece of the interleaved update
            t2 = time.perf_counter()
            self._poll()
            acc["t_learner"] += time.perf_counter() - t2

    def _checkpoint(self, pending: List[_Group], acc: Dict, need=None) -> None:
        """A checkpoint of ``post_ahead``: posts the pending groups (launched, not posted) whose step has finished
        or whose deadline has come (waiting for it), and ``need`` (a group whose workers are about to be waited
        for; "all": every pending group)."""
        i = 0
        while i < len(pending):
            g = pending[i]
            g.deadline -= 1
            if need == "all" or g is need or g.deadline <= 0 or g.ready():
                del pending[i]
                self._post(g, acc)
            else:
                i += 1

    @torch.no_grad()
    def collect(self, ro: Rollout, gae: bool = True, poll=None) -> Dict:
        """Fills ``ro`` (one of ``self.rollouts``) with T steps; stores the bootstrap value (``ro.next_value``,
        ``ro.next_done``) and, with ``gae``, computes the advantages/returns (else :func:`compute_advantages` must be
        called before the update). ``poll`` (e.g. ``Learner.poll``) is called after every post. Returns rollout
        statistics (raw reward per term and step, finished lives, action histogram, timing)."""
        self._poll = poll
        self.posted_steps = 0  # env-steps posted so far in this rollout (for the caller's poll)
        assert ro is self.rollouts[ro.index]
        T, base = ro.T, ro.index * ro.T
        if self.store.inj_pool is not None:
            self.store.inj_pool.begin_rollout()
        cuda = self.stream is not None
        if cuda:
            self.stream.wait_stream(torch.cuda.current_stream())
        ev = self._rew_event[ro.index]
        if ev is not None:
            ev.synchronize()  # the previous upload of this host reward buffer is done
        rew = self._rew_host[ro.index].numpy()
        acc = {"raw": 0.0, "terms": np.zeros(len(TERMS), dtype=np.float64), "waiting": 0,
               "act_hist": np.zeros(NUM_ACTIONS, dtype=np.int64), "episodes": [], "t_host": 0.0, "t_act": 0.0,
               "t_act_wait": 0.0, "t_learner": 0.0, "t_reset_wait": 0.0}
        for g in self.groups:
            g.env.pop_timing()
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream) if cuda else contextlib.nullcontext():
            for g in self.groups:
                ro.h0[:, g.cols].copy_(g.step.h)
        for g in self.groups:
            self._act_post(g, base, 0, acc)
        if not self.async_reset:
            for t in range(1, T):
                for g in self.groups:
                    self._host(g, rew, acc)
                    self._act_post(g, base + t, t, acc)
            for g in self.groups:
                self._host(g, rew, acc)
        else:
            # Groups in turn as long as none resets. A group whose step ended lives resets them in a helper thread
            # (~8.7 ms of worker time per new-seed bedroom start) while the other groups keep stepping (they may run
            # ahead within the rollout; every group does exactly T steps). With post_ahead, a launched policy step
            # is posted at a later checkpoint (class docstring).
            active = list(self.groups)
            pending: List[_Group] = []
            ahead = self.post_ahead
            while active:
                progressed = False
                for g in list(active):
                    if not g.begun:
                        if ahead:
                            self._checkpoint(pending, acc, need=g)
                        g.env.step_wait_begin(async_reset=True)
                        g.begun = True
                        if ahead:
                            self._checkpoint(pending, acc)
                    if g.env.reset_ready():
                        self._host(g, rew, acc)
                        progressed = True
                        if ahead:
                            self._checkpoint(pending, acc)
                        if g.t + 1 < T:
                            if ahead:
                                self._launch(g, base + g.t + 1, g.t + 1, acc)
                                pending.append(g)
                            else:
                                self._act_post(g, base + g.t + 1, g.t + 1, acc)
                        else:
                            active.remove(g)
                if not progressed:  # every active group is waiting for a reset
                    self._checkpoint(pending, acc, need="all")  # (their workers step meanwhile)
                    tw = time.perf_counter()
                    futures.wait([g.env.reset_future() for g in active], return_when=futures.FIRST_COMPLETED)
                    acc["t_reset_wait"] += time.perf_counter() - tw
            assert not pending  # (a group's last step is posted before its step_wait_begin)
        # bootstrap: one more policy step on the next observation, stored in the scratch row; the
        # recurrent state is restored (the next rollout's first step acts on the same observation)
        t1 = time.perf_counter()
        sc = self.store.scratch
        with torch.cuda.stream(self.stream) if cuda else contextlib.nullcontext():
            for g in self.groups:
                g.h_save.copy_(g.step.h)
            for g in self.groups:
                g.act(sc, self.stream)
            for g in self.groups:
                g.step.h.copy_(g.h_save)
            ro.next_value.copy_(self.store.fields["values"][sc])
            ro.next_done.copy_(self.store.fields["done"][sc])
            ro.rewards.copy_(self._rew_host[ro.index], non_blocking=True)
            if ev is not None:
                ev.record()
            if gae:
                compute_advantages(ro, self.cfg)
        if cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        t_end = time.perf_counter()
        steps = T * self.N
        timing = {"act": acc["t_act"], "act_wait": acc["t_act_wait"], "host": acc["t_host"], "bootstrap": t_end - t1}
        if self.async_reset:
            timing["reset_wait"] = acc["t_reset_wait"]
        if poll is not None:
            timing["learner"] = acc["t_learner"]
        self._poll = None
        env_s = 0.0
        for g in self.groups:
            for k, v in g.env.pop_timing().items():
                timing[k] = timing.get(k, 0.0) + v
                env_s += v
        return {"seconds": t_end - t0, "env_seconds": env_s, "steps": steps, "timing": timing,
                "reward_raw": acc["raw"] / steps, "terms": {k: acc["terms"][j] / steps for j, k in enumerate(TERMS)},
                "waiting_frac": acc["waiting"] / steps, "actions": acc["act_hist"] / steps,
                "episodes": acc["episodes"]}


def compute_advantages(ro: Rollout, cfg: PPOConfig) -> None:
    """GAE of a collected rollout from its stored bootstrap value (fills ``advantages`` / ``returns``)."""
    adv, ret = compute_gae(ro.rewards, ro.values, ro.done, ro.next_value, ro.next_done, cfg.gamma, cfg.gae_lambda)
    ro.advantages.copy_(adv)
    ro.returns.copy_(ret)


def compute_gae(rewards, values, done, next_value, next_done, gamma: float, lam: float):
    """GAE over [T, N]: ``done[t]`` marks obs t as a life's start (CleanRL); ``next_*`` the same for the observation
    after step T - 1. Returns (advantages, returns = advantages + values). Per step t, with b = done[t + 1]
    (``next_done`` at t = T - 1) and v' = values[t + 1] (``next_value``): delta_t = r_t + gamma (1 - b) v' - values[t],
    A_t = delta_t + gamma lam (1 - b) A_{t+1}: every life end, a truncation too, is a terminal state of value 0.
    Graph-friendly: static shapes, no host sync."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            nonterm, nv = 1.0 - next_done, next_value
        else:
            nonterm, nv = 1.0 - done[t + 1], values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def minibatch_loss(model: Policy, obs, h0, done, mask, actions, logprobs, advantages, returns, idx, ent_coef,
                   clip_eps: float, vf_coef: float, amp: bool, inj=None, h_inj=None, inj_entry=None):
    """PPO loss of the env sequences ``idx`` (columns of the rollout tensors): returns (loss, pg_loss, v_loss,
    entropy, approx_kl, clipfrac). The forward pass runs under bf16 autocast when ``amp``; logits and values
    are cast back to fp32 before the losses. Free of data-dependent shapes so that the whole function
    (gathers, forward, losses, and through AOTAutograd its backward) compiles to one graph. ``inj`` / ``h_inj``:
    the rollout's rewind injections (dense, sliced like ``done``); with ``inj_entry``, ``h_inj`` is the pool to gather
    the minibatch's states from (:func:`_injections`)."""
    obs = {k: v[:, idx] for k, v in obs.items()}
    mask = mask[:, idx]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        if inj is None:
            logits, values = model.sequence(obs, h0[:, idx], done[:, idx], mask)
        else:
            h_i = h_inj[:, idx] if inj_entry is None else h_inj[inj_entry[:, idx]]
            logits, values = model.sequence(obs, h0[:, idx], done[:, idx], mask, inj[:, idx], h_i)
    logits, values = logits.float(), values.float()
    logp_all = torch.log_softmax(logits, dim=-1)
    newlogp = logp_all.gather(-1, actions[:, idx][..., None]).squeeze(-1)
    # Policy terms are averaged over DECISION steps only (>= 2 allowed actions): on a step where the mask
    # leaves a single action (NOOP while the game is busy, or a forced A) the policy has nothing to choose,
    # its log-prob is 0 and ratio 1, so counting those steps would scale the policy gradient, the entropy
    # pressure and the KL/clip statistics by the waiting fraction. The value loss uses every step.
    dec = (mask.sum(-1) > 1).to(logp_all.dtype)
    n_dec = dec.sum().clamp(min=1.0)
    entropy = (masked_entropy(logp_all, mask) * dec).sum() / n_dec
    logratio = newlogp - logprobs[:, idx]
    ratio = logratio.exp()
    with torch.no_grad():
        approx_kl = (((ratio - 1) - logratio) * dec).sum() / n_dec
        clipfrac = (((ratio - 1.0).abs() > clip_eps).to(dec.dtype) * dec).sum() / n_dec
    # advantages normalised by the mean and (unbiased) std over the decision steps
    adv = advantages[:, idx]
    a_mean = (adv * dec).sum() / n_dec
    a_std = ((((adv - a_mean) * dec) ** 2).sum() / (n_dec - 1.0).clamp(min=1.0)).sqrt()
    adv = (adv - a_mean) / (a_std + 1e-8)
    pg_loss = (torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip_eps, 1 + clip_eps)) * dec).sum() / n_dec
    v_loss = ((values - returns[:, idx]) ** 2).mean()
    loss = pg_loss - ent_coef * entropy + vf_coef * v_loss
    return loss, pg_loss, v_loss, entropy, approx_kl, clipfrac


def _injections(ro) -> tuple:
    """The rollout's rewind fields as extra loss arguments (none when the rewind is off)."""
    inj = getattr(ro, "inj", None)
    if inj is None:
        return ()
    return (inj, ro.h_inj) if ro.inj_entry is None else (inj, ro.inj_pool, ro.inj_entry)


def ppo_update(model: Policy, opt: torch.optim.Optimizer, ro: Rollout, cfg: PPOConfig) -> Dict:
    """``epochs`` x ``minibatches`` of whole env sequences. Returns loss statistics. The forward pass runs under
    bf16 autocast when ``cfg.autocast`` (on CUDA); logits and values are cast back to fp32 before the losses."""
    it = ppo_update_steps(model, opt, ro, cfg)
    while True:
        try:
            next(it)
        except StopIteration as e:
            return e.value


STAT_KEYS = ("pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac", "grad_norm")


class UpdateGraphs:
    """The learner's minibatch step (compiled loss, backward, grad-norm clip, fused Adam step, statistics, the
    approx KL to pinned host memory) and GAE as CUDA graphs, one pair per rollout buffer (the rollout tensors
    are the graphs' fixed inputs; the buffers alternate). Host time per minibatch: ~0.03 ms instead of ~5 ms
    of kernel launches. Same computation and order as the eager loop: the minibatch indices are copied into
    ``idx`` before each replay (``torch.randperm`` per epoch, outside the graph, as before), ``ent_coef`` is filled
    at the start of each update, the gradients are produced and consumed inside each replay (``set_to_none``
    semantics), and the KL stop still reads each minibatch's approx KL on the host before the next replay.

    Capture (first update on a rollout buffer, :meth:`prepare`): two warm-up steps run eagerly, the optimiser's
    param groups are switched to ``capturable=True`` (the fused Adam keeps ``step`` on the device either way and
    its arithmetic is unchanged), the graphs are captured, and the parameters and optimiser state are then restored
    in place (a fresh optimiser state is zeroed: identical to a lazily initialised one), so capturing does not change
    the training. The parameters, gradients and optimiser state are read and written in place: do not
    replace them (``load_state_dict`` of the model is in place; of the optimiser it is not: build a new
    ``UpdateGraphs`` after loading an optimiser state). No ``torch.cuda.empty_cache()`` while graphs live."""

    def __init__(self, model: Policy, opt: torch.optim.Optimizer, cfg: PPOConfig, device):
        self.model, self.opt, self.cfg = model, opt, cfg
        self.device = torch.device(device)
        self.params = [p for p in model.parameters()]
        self.ent_coef = torch.zeros((), device=self.device)
        self.slot = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.kl_host = torch.zeros((), pin_memory=True)
        self.idx = None
        self.stats = None
        self.pool = torch.cuda.graph_pool_handle()
        self._graphs: Dict[int, tuple] = {}
        # the stream to warm up and capture on (Learner: its own stream); None: a new torch.cuda.Stream() per
        # buffer, which can be the CUDA stream other graphs were captured on (StepGraph: shared cuBLAS workspace)
        self.capture_stream: Optional[torch.cuda.Stream] = None

    def _mb_body(self, ro: Rollout) -> None:
        cfg = self.cfg
        loss_fn = getattr(self.model, "ppo_loss", minibatch_loss)  # compiled by prepare_learner
        loss, pg_loss, v_loss, entropy, approx_kl, clipfrac = loss_fn(
            self.model, ro.obs, ro.h0, ro.done, ro.mask, ro.actions, ro.logprobs, ro.advantages, ro.returns,
            self.idx, self.ent_coef, cfg.clip_eps, cfg.vf_coef, bool(cfg.autocast), *_injections(ro))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
        self.opt.step()
        row = torch.stack([pg_loss.detach(), v_loss.detach(), entropy.detach(), approx_kl, clipfrac, gn.detach()])
        self.stats.index_copy_(0, self.slot, row[None].float())
        self.slot.add_(1)
        self.kl_host.copy_(approx_kl, non_blocking=True)

    def _capture(self, fn, stream) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        # (not torch.cuda.graph(): it calls empty_cache(), which must not run while the step graphs live)
        with torch.cuda.stream(stream):
            g.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                fn()
            finally:
                g.capture_end()
        return g

    def prepare(self, ro: Rollout) -> None:
        """Captures the graphs of rollout buffer ``ro`` if needed (on the current stream's work order)."""
        if id(ro) in self._graphs:
            return
        opt, cfg = self.opt, self.cfg
        mb = max(1, ro.N // cfg.minibatches)
        if self.idx is None:
            self.idx = torch.arange(mb, device=self.device)
            self.stats = torch.zeros(cfg.epochs * -(-ro.N // mb), len(STAT_KEYS), device=self.device)
        assert self.idx.numel() == mb
        saved = [p.detach().clone() for p in self.params]
        saved_state = {p: {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state[p].items()}
                       for p in self.params if p in opt.state}
        cur = torch.cuda.current_stream(self.device)
        side = self.capture_stream or torch.cuda.Stream(device=self.device)
        side.wait_stream(cur)
        with torch.cuda.stream(side), torch.enable_grad():
            self.idx.copy_(torch.arange(mb, device=self.device))
            self.ent_coef.fill_(cfg.ent_coef)
            for _ in range(2):  # warm-up: lazy initialisation (optimiser state, autograd, cuBLAS) outside capture
                self.slot.zero_()
                opt.zero_grad(set_to_none=True)
                self._mb_body(ro)
            compute_advantages(ro, cfg)
            for gr in opt.param_groups:  # (after the warm-up: no "capturable but not capturing" warning)
                gr["capturable"] = True
            opt.zero_grad(set_to_none=True)  # the captured backward allocates the gradients in the pool
            g_mb = self._capture(lambda: self._mb_body(ro), side)
            g_gae = self._capture(lambda: compute_advantages(ro, cfg), side)
            with torch.no_grad():  # undo the warm-up steps
                torch._foreach_copy_(self.params, saved)
                for p in self.params:
                    st = opt.state[p]
                    old = saved_state.get(p)
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            v.copy_(old[k]) if old is not None else v.zero_()
            self.slot.zero_()
        cur.wait_stream(side)
        self._graphs[id(ro)] = (ro, g_mb, g_gae)

    def gae(self, ro: Rollout) -> None:
        self.prepare(ro)
        self._graphs[id(ro)][2].replay()

    def begin(self, ro: Rollout) -> None:
        """Before the first minibatch of an update on ``ro`` (current stream)."""
        self.prepare(ro)
        self.ent_coef.fill_(self.cfg.ent_coef)
        self.slot.zero_()

    def step(self, ro: Rollout, idx: torch.Tensor) -> None:
        self.idx.copy_(idx)
        self._graphs[id(ro)][1].replay()

    def results(self, n_mb: int) -> Dict[str, float]:
        m = self.stats[:n_mb].mean(0).tolist()
        return dict(zip(STAT_KEYS, m))


def ppo_update_steps(model: Policy, opt: torch.optim.Optimizer, ro: Rollout, cfg: PPOConfig,
                     graphs: Optional[UpdateGraphs] = None):
    """:func:`ppo_update` as a generator: yields after enqueuing each minibatch (a CUDA event recorded after
    it, or None on the CPU) and returns the statistics. The KL stop reads the minibatch's approx KL only
    when resumed (after waiting for the event), so a caller that resumes once ``event.query()`` is true
    never blocks on the GPU and the update is the same as :func:`ppo_update`'s. With ``graphs``
    (:class:`UpdateGraphs`) each minibatch step is one graph replay."""
    N = ro.N
    cuda = ro.h0.device.type == "cuda"
    kl_host = graphs.kl_host if graphs is not None else torch.zeros((), pin_memory=cuda)
    amp = bool(cfg.autocast) and cuda
    loss_fn = getattr(model, "ppo_loss", minibatch_loss)  # compiled by prepare_learner
    mb = max(1, N // cfg.minibatches)
    if graphs is not None:
        graphs.begin(ro)
    else:
        ent_coef = torch.tensor(cfg.ent_coef, device=ro.h0.device)
    stats = {k: [] for k in STAT_KEYS}
    stopped = False
    n_mb = 0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(N, device=ro.h0.device)
        for s in range(0, N, mb):
            idx = perm[s:s + mb]
            if graphs is not None:
                graphs.step(ro, idx)
            else:
                loss, pg_loss, v_loss, entropy, approx_kl, clipfrac = loss_fn(
                    model, ro.obs, ro.h0, ro.done, ro.mask, ro.actions, ro.logprobs, ro.advantages, ro.returns, idx,
                    ent_coef, cfg.clip_eps, cfg.vf_coef, amp, *_injections(ro))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                for k, v in zip(STAT_KEYS, (pg_loss, v_loss, entropy, approx_kl, clipfrac, gn)):
                    stats[k].append(v.detach())
                kl_host.copy_(approx_kl, non_blocking=True)
            n_mb += 1
            ev = None
            if cuda:
                ev = torch.cuda.Event()
                ev.record()
            yield ev
            if ev is not None:
                ev.synchronize()
            if float(kl_host) > cfg.target_kl:
                stopped = True
                break
        if stopped:
            break
    if graphs is not None:
        out = graphs.results(n_mb)
    else:
        out = {k: float(torch.stack(v).mean()) for k, v in stats.items()}
    with torch.no_grad():
        y, p = ro.returns.flatten(), ro.values.flatten()
        var_y = y.var()
        out["explained_var"] = float(1 - (y - p).var() / var_y) if var_y > 0 else float("nan")
    out["minibatches"] = n_mb
    out["kl_stop"] = float(stopped)
    out["seconds"] = time.perf_counter() - t0
    return out


def prepare_learner(model: Policy, cfg: PPOConfig) -> None:
    """With ``cfg.compile`` (CUDA only): ``model.ppo_loss`` = ``torch.compile(minibatch_loss, dynamic=False)``,
    i.e. the gathers, ``model.sequence`` (inlined), the losses and their backward compile together. The
    compiled function is an instance attribute (``state_dict`` unchanged); make the acting copy
    (``copy.deepcopy``) before calling this. Also sets ``torch.backends.cudnn.benchmark`` (``cfg.cudnn_benchmark``)."""
    if next(model.parameters()).device.type != "cuda":
        return
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if cfg.compile:
        model.ppo_loss = torch.compile(minibatch_loss, dynamic=False)


def warm_up_learner(model: Policy, ro: Rollout, cfg: PPOConfig) -> None:
    """Runs the (compiled) loss and its backward once on a minibatch of ``ro`` without an optimiser step,
    so that compilation happens here and not inside the first overlapped update."""
    mb = max(1, ro.N // cfg.minibatches)
    idx = torch.arange(mb, device=ro.h0.device)
    amp = bool(cfg.autocast) and ro.h0.device.type == "cuda"
    loss_fn = getattr(model, "ppo_loss", minibatch_loss)
    with torch.enable_grad():
        out = loss_fn(model, ro.obs, ro.h0, ro.done, ro.mask, ro.actions, ro.logprobs, ro.advantages, ro.returns, idx,
                      torch.tensor(cfg.ent_coef, device=ro.h0.device), cfg.clip_eps, cfg.vf_coef, amp, *_injections(ro))
        (out[0] * 0).backward()
    model.zero_grad(set_to_none=True)


@torch.no_grad()
def sync_actor(actor: Policy, model: Policy) -> None:
    """Copies the learner's parameters and buffers into the acting copy (in place: the step graphs read
    the actor's tensors), on the current stream."""
    dst = list(actor.parameters()) + list(actor.buffers())
    src = list(model.parameters()) + list(model.buffers())
    torch._foreach_copy_(dst, src)


class Learner:
    """The PPO update of one rollout overlapped with the collection of the next, on its own CUDA stream.
    ``start(ro)`` orders the learner stream after the caller's current stream (the collector's writes of ``ro``);
    the update (GAE, then :func:`ppo_update_steps`) then runs

    * ``mode="thread"``: in a background thread (torch releases the GIL inside its ops);
    * ``mode="interleave"``: cooperatively in the caller's thread: ``poll()`` (the collector calls it after
      posting each env group's actions, while the workers simulate) enqueues the next minibatch once the
      previous one has finished on the GPU, and otherwise returns at once.

    ``wait()`` finishes the update, orders the caller's current stream after the learner's work and
    returns the update statistics (``seconds``: wall time from ``start`` to the end of the update;
    ``host_seconds``: time spent enqueuing in ``poll``/``wait``, interleave mode). ``ro`` must not be
    written until ``wait()`` returned."""

    def __init__(self, model: Policy, opt: torch.optim.Optimizer, cfg: PPOConfig, device, mode: str = "thread"):
        if mode not in ("thread", "interleave"):
            raise ValueError(f"Learner mode: thread or interleave, not {mode!r}")
        self.model, self.opt, self.cfg, self.mode = model, opt, cfg, mode
        self.device = torch.device(device)
        cuda = torch.device(device).type == "cuda"
        # low priority (0, the default, is the lowest): the collector's step graphs go first
        self.stream = torch.cuda.Stream(device=device, priority=0) if cuda else None
        # graphs need capturable=True, which changes the arithmetic of the foreach Adam but not of the fused one
        graph = cuda and cfg.graph_update and bool(opt.defaults.get("fused"))
        self.graphs = UpdateGraphs(model, opt, cfg, device) if graph else None
        if self.graphs is not None:
            # captured on the learner's own stream, never the collector's stream, on which the step graphs are
            # captured (distinct cuBLAS workspaces, StepGraph)
            self.graphs.capture_stream = self.stream
        self._thread: Optional[threading.Thread] = None
        self._gen = None
        self._ev = None
        self._result = None
        self._started = False
        self._host = 0.0

    def _ctx(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()

    def set_lr(self, lr: float) -> None:
        """The learning rate of the next updates (call while no update runs: before ``start`` / after ``wait``).
        Without graphs: the param groups' float. With the CUDA-graphed learner the fused Adam passes a float lr
        to its kernel as a scalar argument, i.e. a constant baked into the captured graphs (changing the float
        between replays would do nothing); a CUDA tensor lr is read by every replay instead. So a lr other than
        ``cfg.lr`` is held in a device tensor (the graphs are re-captured once when the float was in use: at the
        first warm-up update, before anything was captured), and every later change, including the return to
        ``cfg.lr`` at the end of the warm-up, is a ``fill_``."""
        groups = self.opt.param_groups
        lr = float(lr)
        if self.graphs is None:
            for gr in groups:
                gr["lr"] = lr
            return
        tensor = any(torch.is_tensor(gr["lr"]) for gr in groups)
        if not tensor and all(gr["lr"] == lr for gr in groups):
            return
        if not tensor:
            for gr in groups:
                gr["lr"] = torch.tensor(lr, dtype=torch.float32, device=self.device)
            self._new_graphs()
        else:
            for gr in groups:
                gr["lr"].fill_(lr)

    def _new_graphs(self) -> None:
        """Drops the captured update graphs (re-captured by the next update, as on the first one)."""
        if self.graphs is None:
            return
        old, self.graphs = self.graphs, None
        if old._graphs:
            for gr in self.opt.param_groups:  # (as before the first capture: prepare sets it after its warm-up)
                gr["capturable"] = False
            # free the old graphs' private pool before the re-capture: the gradients live in it, and the cached old
            # pool next to the new one doubled the update's memory
            self.opt.zero_grad(set_to_none=True)
            old._graphs.clear()
            del old
            gc.collect()
            torch.cuda.empty_cache()
        self.graphs = UpdateGraphs(self.model, self.opt, self.cfg, self.device)
        self.graphs.capture_stream = self.stream

    def _steps(self, ro: Rollout):
        if self.graphs is not None:
            self.graphs.gae(ro)
        else:
            compute_advantages(ro, self.cfg)
        return (yield from ppo_update_steps(self.model, self.opt, ro, self.cfg, graphs=self.graphs))

    def _run(self, ro: Rollout) -> None:
        try:
            with self._ctx():
                it = self._steps(ro)
                while True:
                    try:
                        next(it)
                    except StopIteration as e:
                        self._result = e.value
                        break
        except BaseException as e:  # re-raised by wait()
            self._result = e

    def start(self, ro: Rollout) -> None:
        assert not self.busy, "start() twice without wait()"
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())
        if self.graphs is not None:  # capture on the first update of each rollout buffer (in this thread)
            with self._ctx():
                self.graphs.prepare(ro)
        self._result, self._ev, self._host = None, None, 0.0
        self._started = True
        if self.mode == "thread":
            self._thread = threading.Thread(target=self._run, args=(ro,), name="ppo-learner", daemon=True)
            self._thread.start()
        else:
            self._gen = self._steps(ro)

    @property
    def busy(self) -> bool:
        """Started and not yet waited for."""
        return self._started

    @property
    def finished(self) -> bool:
        """Started and the update has finished (``wait()`` will not block on it)."""
        if not self._started:
            return False
        if self._thread is not None:
            return not self._thread.is_alive()
        return self._gen is None

    def _advance(self) -> None:
        t0 = time.perf_counter()
        try:
            with self._ctx(), torch.enable_grad():  # (poll runs inside the collector's no_grad)
                self._ev = next(self._gen)
        except StopIteration as e:
            self._result, self._gen = e.value, None
        except BaseException as e:
            self._result, self._gen = e, None
        self._host += time.perf_counter() - t0

    def poll(self) -> None:
        """Interleave mode: enqueue the next piece of the update if the GPU finished the previous one."""
        if self._gen is not None and (self._ev is None or self._ev.query()):
            self._advance()

    def wait(self) -> Dict:
        assert self.busy, "wait() without start()"
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        while self._gen is not None:
            self._advance()
        self._started = False
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        r, self._result = self._result, None
        if isinstance(r, BaseException):
            raise r
        r["host_seconds"] = self._host
        return r
