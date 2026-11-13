"""The trainer end to end on the CPU (a few envs, short rollouts): a fresh run, its checkpoint, the logs and a
resume."""
import csv
import json
import os

import torch

from conftest import needs_worker
from pokerl import train as TR

SMALL = ["--device", "cpu", "--collector", "sync", "--no-compile", "--no-autocast", "--num-envs", "4", "--rollout",
         "16", "--minibatches", "2", "--num-workers", "2", "--checkpoint-every", "1", "--stagnation-steps", "64",
         "--rewind-min-gap", "8", "--max-episode-steps", "200", "--stagger-min", "100", "--stagger-max", "200"]


@needs_worker
def test_fresh_run_and_resume(tmp_path):
    torch.set_num_threads(2)
    runs = str(tmp_path)
    row = TR.main(SMALL + ["--runs-dir", runs, "--run-name", "smoke", "--total-steps", "128"])
    run = os.path.join(runs, "smoke")
    assert row["update"] == 2 and row["global_step"] == 128
    cfg = json.load(open(os.path.join(run, "config.json")))
    assert cfg["num_envs"] == 4 and cfg["ppo"]["gamma"] == 0.997 and cfg["relay_activation_reward"] == 2.0
    ck = torch.load(os.path.join(run, "ckpt_000002.pt"), map_location="cpu", weights_only=False)
    assert ck["update"] == 2 and "tile_count_idx" in ck["reward"]
    rows = list(csv.DictReader(open(os.path.join(run, "log.csv"))))
    assert [r["update"] for r in rows] == ["1", "2"] and all(float(r["lr"]) < 2.5e-4 for r in rows)
    row = TR.main(SMALL + ["--runs-dir", runs, "--resume", "smoke", "--total-steps", "192"])
    assert row["update"] == 3 and os.path.exists(os.path.join(run, "ckpt_000003.pt"))


@needs_worker
def test_pipelined_collector_and_learner_on_the_cpu(tmp_path):
    torch.set_num_threads(2)
    args = [a for a in SMALL if a not in ("--collector", "sync")]
    row = TR.main(args + ["--collector", "pipelined", "--env-groups", "2", "--runs-dir", str(tmp_path), "--run-name",
                          "pipe", "--total-steps", "192"])
    assert row["update"] == 3 and 0.0 <= row["policy_lag"] <= 1.0 and row["ep_count"] >= 0
