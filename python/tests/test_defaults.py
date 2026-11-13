"""Default worker counts (python/README.md "Workers")."""
from __future__ import annotations

from fireredsim.vec_env import MAX_PROCESSES_PER_CPU, default_num_workers, spin_worker_limit


def test_default_num_workers_32_cpus():
    d = {n: default_num_workers(n, 32) for n in (1, 16, 24, 25, 28, 32, 40, 48, 49, 64, 128, 129, 256, 512, 513,
                                                 1000)}
    assert spin_worker_limit(32) == 24
    # one env per busy-waiting worker up to 3/4 of the CPUs
    assert d[1] == 1 and d[16] == 16 and d[24] == 24
    # two envs per worker (still <= 24 busy-waiting workers) instead of ~one futex worker per vCPU
    assert d[25] == 13 and d[28] == 14 and d[32] == 16 and d[40] == 20 and d[48] == 24
    # oversubscribed, one env per worker, up to 16 per CPU (packing many workers is memory-bandwidth bound)
    assert MAX_PROCESSES_PER_CPU == 16
    assert d[49] == 49 and d[64] == 64 and d[128] == 128 and d[129] == 129 and d[256] == 256 and d[512] == 512
    assert d[513] == d[1000] == MAX_PROCESSES_PER_CPU * 32


def test_default_num_workers_small_machines():
    for cpus in (1, 2, 3, 4, 8, 12, 16):
        cap = spin_worker_limit(cpus)
        for n in range(1, 6 * cpus + 2):
            w = default_num_workers(n, cpus)
            assert 1 <= w <= n
            if n <= 2 * cap:
                assert w <= cap  # busy-waiting workers fit
                assert -(-n // w) <= 2  # at most two envs per worker
