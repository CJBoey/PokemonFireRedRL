"""Decision-point stepping (``PokeVecEnv(auto_advance=True)``, PnEnvCmd flag PN_F_AUTO_ADVANCE).

Exactness: an env with auto_advance returns, after each action, exactly what an env without it
returns when it is fed the action and then NOOP steps of ``advance_frames`` frames (the last one
shortened to the cap) up to the first observation with ``waiting_for_input == 1``: every
observation byte, the state hash and the frames run agree. Random actions from the bedroom start, several hundred decision steps on 12 envs, chunked (the default: the step's own 24 frames)
and per-frame (``advance_frames=1``) advancing, small and default caps, with and without
rng_reseed. Also: the cap / non-waiting rule, per-call overrides, a worker without the flag.
Observer v6's interaction fields report every frame of the op: an auto-advanced step's are the
reference records' combined (the first interaction of the requested step and its NOOP steps, the
counts summed).
"""
import numpy as np
import pytest

import fireredsim
from fireredsim import _ipc


# NOOP, directions, A, B, START, SELECT (KEYINPUT, active high)
BUTTONS = np.array([0x000, 0x010, 0x020, 0x040, 0x080, 0x001, 0x002, 0x008, 0x004], dtype=np.uint16)
P = np.array([0.04, 0.15, 0.15, 0.15, 0.15, 0.24, 0.08, 0.03, 0.01])


def _skill_bytes():
    """Byte range of PnObs.skill_events_priv (observer v5) or None."""
    from fireredsim import obs_dtype as od
    f = od.OBS_DTYPE.fields.get("skill_events_priv")
    return None if f is None else slice(f[1], f[1] + f[0].itemsize)


SKILL = _skill_bytes()


def _interact_bytes():
    """Byte range of observer v6's interact_*_priv fields (pads included) or None."""
    from fireredsim import obs_dtype as od
    f = od.OBS_DTYPE.fields.get("interact_kind_priv")
    v7 = od.OBS_DTYPE.fields.get("offered_move")  # observer v7 appends after the v6 block
    return None if f is None else slice(f[1], od.OBS_DTYPE.itemsize if v7 is None else v7[1])


INTERACT = _interact_bytes()


def _learned_field():
    """Observer v7's learned_move_priv (u16[6]) is present."""
    from fireredsim import obs_dtype as od
    return "learned_move_priv" in od.OBS_DTYPE.names


LEARNED = _learned_field()


def _env(worker_path, n, start, **kw):
    return fireredsim.PokeVecEnv(num_envs=n, num_workers=n, frameskip=24, hold_frames=1, obs_mode="privileged",
                                 action_type="buttons", start=start, seed=5, worker_path=worker_path, **kw)


def _ref_step(ref, actions, cap, chunk):
    """The reference: the action's 24-frame step, then NOOP steps of ``chunk`` frames (shortened to the
    cap) on every env whose observation is not waiting. Returns (obs bytes, frames, waiting). The v6
    interaction bytes of the result are those of all the env's records combined: the first record that
    reports an interaction, with the counts of all of them summed (saturating at 255). Observer v7's
    learned_move_priv likewise: per party slot, the first nonzero report of the env's records (the worker compares
    the auto-advanced record with the previous step's record, the reference each chunk's with the chunk before)."""
    n = ref.num_envs
    ob = np.zeros((n, ref.obs_dtype.itemsize), dtype=np.uint8)
    first = np.zeros((n, 0 if INTERACT is None else INTERACT.stop - INTERACT.start), dtype=np.uint8)
    learned = np.zeros((n, 6), dtype=np.uint16)
    count = np.zeros(n, dtype=np.int64)
    frames = np.zeros(n, dtype=np.int64)
    stop = np.zeros(n, dtype=bool)  # soft reset / terminal: the op ends there
    waiting = np.zeros(n, dtype=bool)

    def take(obs, info):
        ids = info["env_id"]
        ob[ids] = np.asarray(obs).view(np.uint8).reshape(ids.size, -1)
        frames[ids] += info["frames_run"]
        if INTERACT is not None:
            c = np.asarray(obs["interact_count_priv"]).astype(np.int64)
            new = ids[(c > 0) & (count[ids] == 0)]
            first[new] = ob[new, INTERACT]
            count[ids] += c
        if LEARNED:
            lm = np.asarray(obs["learned_move_priv"]).astype(np.uint16)
            learned[ids] = np.where(learned[ids] != 0, learned[ids], lm)
        stop[ids] |= info["soft_reset"] | info["sim_terminal"]
        waiting[ids] = np.asarray(obs["waiting_for_input"]) != 0

    ref.send(actions)
    r = ref.recv()
    take(r[0], r[4])
    while True:
        pend = np.flatnonzero(~waiting & ~stop & (frames < cap))
        if pend.size == 0:
            break
        ks = np.minimum(chunk, cap - frames[pend])
        for k in np.unique(ks):
            ids = pend[ks == k]
            ref.send(np.zeros(ids.size, dtype=np.uint16), env_ids=ids, n_frames=int(k), hold_frames=0)
        while ref._async["inflight"].any():
            r = ref.recv()
            take(r[0], r[4])
    if INTERACT is not None:
        ob[:, INTERACT] = first
        ob[:, INTERACT.start + 1] = np.minimum(count, 255)  # interact_count_priv
    if LEARNED:
        from fireredsim import obs_dtype as od
        off = od.OBS_DTYPE.fields["learned_move_priv"][1]
        ob[:, off:off + 12] = learned.astype("<u2").view(np.uint8).reshape(n, 12)
    return ob, frames, waiting


def _run_equivalence(worker_path, start, steps, cap, chunk, rng_reseed=False, small_cap_every=0, n=12, seed=0):
    rng = np.random.default_rng(seed)
    kw = {"rng_reseed": rng_reseed}
    with _env(worker_path, n, start, auto_advance=True, max_frames=cap, advance_frames=chunk or None, **kw) as aa, \
            _env(worker_path, n, start, **kw) as ref:
        o1, _ = aa.reset()
        o2, _ = ref.reset()
        assert np.array_equal(np.asarray(o1).view(np.uint8), np.asarray(o2).view(np.uint8))
        total = np.zeros(n, dtype=np.int64)
        capped = advanced = interactions = 0
        for t in range(steps):
            a = BUTTONS[rng.choice(len(BUTTONS), size=n, p=P)]
            c = cap
            if small_cap_every and t % small_cap_every == small_cap_every - 1:
                c = 30  # the cap case: a non-waiting observation returned at 30 frames
            obs, _, _, _, info = aa.step(a, max_frames=c)
            ob_ref, fr_ref, w_ref = _ref_step(ref, a, c, chunk or 24)
            got = np.asarray(obs).view(np.uint8).reshape(n, -1)
            if SKILL is not None:  # observer v5: the worker's skill events compare the records an env returned (one
                got, ob_ref = got.copy(), ob_ref.copy()  # per step here, the NOOP steps' records too in the
                got[:, SKILL] = 0                         # reference): not part of the equivalence
                ob_ref[:, SKILL] = 0
            for i in range(n):
                assert np.array_equal(got[i], ob_ref[i]), f"step {t} env {i}: observation differs"
            assert np.array_equal(info["frames_run"].astype(np.int64), fr_ref), f"step {t}: frames_run"
            assert np.array_equal(aa.state_hash(), ref.state_hash()), f"step {t}: state hash"
            w = np.asarray(obs["waiting_for_input"]) != 0
            assert np.array_equal(w, w_ref)
            # non-waiting only at the cap or after a soft reset / terminal
            assert np.all(w | (info["frames_run"] >= c) | info["soft_reset"] | info["sim_terminal"])
            capped += int((~w).sum())
            advanced += int((info["frames_run"] > 24).sum())
            if INTERACT is not None:
                interactions += int((np.asarray(obs["interact_count_priv"]) > 0).sum())
            total += info["frames_run"]
        assert advanced > 0, "no step was auto-advanced: the test exercised nothing"
        return total, capped, advanced, interactions


def test_exact_chunked(worker_path):
    _, capped, advanced, interactions = _run_equivalence(worker_path, "bedroom", 300, 384, 0, small_cap_every=7)
    assert capped > 0  # the small-cap steps returned non-waiting observations
    if INTERACT is not None:
        assert interactions > 0  # the bedroom's NES / PC / sign: the v6 combination was exercised


def test_exact_per_frame(worker_path):
    _run_equivalence(worker_path, "bedroom", 200, 384, 1, small_cap_every=9, n=8, seed=1)


def test_exact_rng_reseed(worker_path):
    _run_equivalence(worker_path, "bedroom", 150, 384, 0, rng_reseed=True, n=8, seed=2)


def test_override_and_default_off(worker_path):
    """auto_advance=False by default (frames_run == frameskip); the per-call override turns it on and
    the next plain call is plain again; send/recv honour it as well."""
    rng = np.random.default_rng(3)
    with _env(worker_path, 4, "bedroom") as env:
        assert env.auto_advance is False
        env.reset()
        for _ in range(40):
            _, _, _, _, info = env.step(BUTTONS[rng.choice(len(BUTTONS), size=4, p=P)])
            assert np.all(info["frames_run"] == 24)
        seen = 0
        for _ in range(40):
            obs, _, _, _, info = env.step(BUTTONS[rng.choice(len(BUTTONS), size=4, p=P)], auto_advance=True,
                                          max_frames=240)
            assert np.all((obs["waiting_for_input"] != 0) | (info["frames_run"] >= 240))
            seen += int((info["frames_run"] > 24).sum())
        env.send(np.zeros(4, dtype=np.uint16), auto_advance=True, max_frames=100)
        obs, _, _, _, info = env.recv()
        assert np.all((obs["waiting_for_input"] != 0) | (info["frames_run"] >= 100))
        env.send(np.zeros(4, dtype=np.uint16))
        _, _, _, _, info = env.recv()
        assert np.all(info["frames_run"] == 24)
        assert seen > 0


def test_layout_and_refusal(worker_path, monkeypatch):
    info = _ipc.worker_info(worker_path)
    _ipc.check_layout(info)
    assert _ipc.supports_auto_advance(info)
    assert info["structs"]["PnEnvCmd"]["fields"]["max_frames"] == [56, 2]
    assert info["structs"]["PnEnvCmd"]["fields"]["advance_frames"] == [58, 2]
    old = dict(info, structs=dict(info["structs"]))
    fields = {k: v for k, v in info["structs"]["PnEnvCmd"]["fields"].items()
              if k not in ("max_frames", "advance_frames")}
    old["structs"]["PnEnvCmd"] = dict(info["structs"]["PnEnvCmd"], fields=fields)
    assert not _ipc.supports_auto_advance(old)
    _ipc.check_layout(old)  # an old worker still works without the option
    monkeypatch.setattr(_ipc, "worker_info", lambda path: old)
    with pytest.raises(RuntimeError, match="auto_advance=True"):
        fireredsim.PokeVecEnv(num_envs=1, worker_path=worker_path, auto_advance=True)
    with pytest.raises(ValueError):
        fireredsim.PokeVecEnv(num_envs=1, worker_path=worker_path, max_frames=70000)
