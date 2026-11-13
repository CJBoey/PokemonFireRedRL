"""Zero-copy observations retain their shared memory after the environment closes."""
import os
import subprocess
import sys

import pytest

import fireredsim


@pytest.mark.parametrize("pixels", ["False", "True", "'only'"])
def test_zero_copy_observation_outlives_environment(worker_path, pixels):
    # Run in a subprocess: a dangling NumPy view segfaults on access rather than
    # raising a Python exception, and must not take down the rest of the suite.
    script = """
import gc
import resource
import sys
import weakref

import fireredsim as pn

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
env = pn.PokeVecEnv(num_envs=1, start="power_on", copy=False,
                    pixels=PIXELS, render=False, worker_path=sys.argv[1])
obs, _ = env.reset()
views = list(obs.values()) if isinstance(obs, dict) else [obs]
# Retain derived views too, as training code commonly slices observations.
views = [view.reshape(-1)[::2] for view in views]
expected = [view.tobytes() for view in views]
mapping_ref = weakref.ref(env._ws._mm)
env.close()
del env, obs
gc.collect()
assert mapping_ref() is not None, "observation lost its memory owner"
assert [view.tobytes() for view in views] == expected
del views
gc.collect()
assert mapping_ref() is None, "shared memory leaked after the last view was released"
""".replace("PIXELS", pixels)
    proc_env = dict(os.environ)
    package_root = os.path.dirname(os.path.dirname(fireredsim.__file__))
    proc_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [package_root, proc_env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-c", script, worker_path], env=proc_env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"exit {result.returncode}: {result.stdout}{result.stderr}"
