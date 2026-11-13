"""Policy/value network: the screen, the trail plane and the previous action.

    pixels uint8 [80, 120] (gray2: the luma averaged over 2 x 2 pixel blocks) / 255 -> pad to [88, 120]
        -> Conv(1->16, 4, s4) -> Conv(16->32, 4, s2, p1) (32 x 11 x 15) ‖ trail uint8 [11, 15] / 255 as a 33rd channel
        -> Conv(33->32, 3, p1) -> Conv(32->16, 1) (ReLU each) -> Linear 16 x 11 x 15 -> pixel_feat (512) + ReLU
    ‖ previous action one-hot (8; none at a life's start)
    core   -> Linear hidden (512) + ReLU -> LayerNorm -> minGRU -> +residual -> LayerNorm -> minGRU -> +residual
    heads  -> policy Linear(hidden -> 8) (masked logits), value Linear(hidden -> 1)

The frame is padded by 4 gray2 rows top and bottom (half a tile: the game's camera sits half a tile off the map
grid), so that the convolutions' cells land on the 11 x 15 map tiles of the trail window, the player's tile at cell
(5, 7); the strides follow the game's 16 x 16-pixel tiles (8 x 8 in the 2 x 2-binned frame), there is no pooling.

Hidden state: tensor [layers, N, hidden]. Rewind: ``step`` and ``sequence`` take optional injected states (``inj``,
``h_inj``, forwarded to every minGRU layer, mingru.py); without them the computation is unchanged. Orthogonal init
(gain √2 hidden, 0.01 policy, 1.0 value), biases 0, fp32. ``chrono_tau`` > 0: the minGRU gate biases start at
−log τ, τ log-uniform per channel in [1, chrono_tau] (MinGRU.chrono_init).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask import NUM_ACTIONS
from .mingru import MinGRU

NO_ACTION = NUM_ACTIONS  # previous action at a life's start
SCREEN_SHAPE = (80, 120)  # the frames (env.PIXEL_SHAPE)
TRAIL_SHAPE = (11, 15)  # the trail window (reward.SEEN_ROWS x SEEN_COLS): the map tiles around the player
SCREEN_PAD = 4  # gray2 rows added top and bottom: half a map tile, so the cells land on the trail's tiles
SCREEN_PLANES_AT = 4  # the trail joins the screen convolutions' input at this module (conv3, after conv2 + ReLU)
TRAIL_SCALE = 1.0 / 255
MASKED_LOGIT = -1e9


def _init(m: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(m.weight, gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, MASKED_LOGIT)


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(masked_logits(logits, mask), dim=-1)


def masked_entropy(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -(logp.exp() * logp).masked_fill(~mask, 0.0).sum(-1)


class Policy(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 2, pixel_feat: int = 512, chrono_tau: float = 0.0,
                 verbose: bool = True):
        super().__init__()
        if pixel_feat < 1:
            raise ValueError(f"Policy: pixel_feat {pixel_feat} must be >= 1")
        self.chrono_tau = float(chrono_tau)
        self.hidden, self.layers = hidden, layers
        self.pixel_shape, self.pixel_feat = SCREEN_SHAPE, pixel_feat
        self.screen_cells = TRAIL_SHAPE
        rows, cols = self.screen_cells
        self.register_buffer("action_ids", torch.arange(NUM_ACTIONS), persistent=False)
        self.register_buffer("plane_scale", torch.tensor([TRAIL_SCALE]).reshape(1, -1, 1, 1), persistent=False)
        self.screen_conv = nn.Sequential(_init(nn.Conv2d(1, 16, 4, stride=4)), nn.ReLU(),
                                         _init(nn.Conv2d(16, 32, 4, stride=2, padding=1)), nn.ReLU(),
                                         _init(nn.Conv2d(32 + 1, 32, 3, padding=1)), nn.ReLU(),
                                         _init(nn.Conv2d(32, 16, 1)), nn.ReLU())
        self.screen_fc = _init(nn.Linear(16 * rows * cols, pixel_feat))
        self.core_in = _init(nn.Linear(pixel_feat + NUM_ACTIONS, hidden))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.grus = nn.ModuleList([MinGRU(hidden) for _ in range(layers)])
        for g in self.grus:
            _init(g.proj)
            if self.chrono_tau > 0:
                g.chrono_init(self.chrono_tau)
        self.pi = _init(nn.Linear(hidden, NUM_ACTIONS), 0.01)
        self.vf = _init(nn.Linear(hidden, 1), 1.0)
        if verbose:
            n = sum(p.numel() for m in (self.screen_conv, self.screen_fc) for p in m.parameters())
            print(f"screen branch: {'x'.join(map(str, SCREEN_SHAPE))} uint8 -> 16/32/32 + trail at conv3 -> 16 per "
                  f"tile ({rows} x {cols}) -> {pixel_feat}, {n:,} parameters")

    def screen_features(self, pixels: torch.Tensor, prev_action: torch.Tensor, trail: torch.Tensor) -> torch.Tensor:
        """uint8 frames [*, 80, 120], int64 [*], uint8 trail [*, 11, 15] -> [*, pixel_feat + NUM_ACTIONS]."""
        lead = pixels.shape[:-2]
        m = int(math.prod(lead)) if lead else 1
        x = pixels.reshape(m, 1, *SCREEN_SHAPE).float() * (1.0 / 255.0)
        x = F.pad(x, (0, 0, SCREEN_PAD, SCREEN_PAD))
        for i, mod in enumerate(self.screen_conv):
            if i == SCREEN_PLANES_AT:
                p = torch.stack([trail.reshape(m, *self.screen_cells)], 1).to(x.dtype) * self.plane_scale
                x = torch.cat([x, p], 1)
            x = mod(x)
        x = F.relu(self.screen_fc(x.flatten(1)))
        pa = (prev_action.reshape(m, 1) == self.action_ids).float()  # (one-hot by comparison: graph capture safe)
        return torch.cat([x, pa], dim=-1).reshape(*lead, -1)

    def _inputs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """The core's input of an observation dict (``pixels``, ``trail``, ``prev_action``)."""
        return self.screen_features(obs["pixels"], obs["prev_action"], obs["trail"])

    # ---- core ----------------------------------------------------------------------------------------

    def initial_state(self, n: int, device=None) -> torch.Tensor:
        return torch.zeros(self.layers, n, self.hidden, device=device)

    def step(self, obs: Dict[str, torch.Tensor], h: torch.Tensor, done: torch.Tensor, mask: torch.Tensor,
             inj: Optional[torch.Tensor] = None,
             h_inj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step for N envs: obs tensors [N, ...], h [layers, N, H], done [N] (1 = obs starts a life), mask
        [N, 8] -> (masked logits [N, 8], value [N], new h). ``inj`` [N] (1 = inject) and ``h_inj`` [layers, N, H]:
        injected states (rewind)."""
        x = F.relu(self.core_in(self._inputs(obs)))
        hs = []
        for i, (norm, gru) in enumerate(zip(self.norms, self.grus)):
            if inj is None:
                hi = gru.step(norm(x), h[i], done)
            else:
                hi = gru.step(norm(x), h[i], done, inj, h_inj[i])
            hs.append(hi)
            x = x + hi
        return masked_logits(self.pi(x), mask), self.vf(x).squeeze(-1), torch.stack(hs)

    def sequence(self, obs: Dict[str, torch.Tensor], h0: torch.Tensor, done: torch.Tensor, mask: torch.Tensor,
                 inj: Optional[torch.Tensor] = None,
                 h_inj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Whole sequences: obs tensors [T, B, ...], h0 [layers, B, H] (state before step 0), done [T, B],
        mask [T, B, 8] -> (masked logits [T, B, 8], value [T, B]) with the parallel minGRU scan. ``inj`` [T, B]
        and ``h_inj`` [T, B, layers, H] (the rollout buffer's layout): injected states (rewind)."""
        x = F.relu(self.core_in(self._inputs(obs)))
        for i, (norm, gru) in enumerate(zip(self.norms, self.grus)):
            if inj is None:
                x = x + gru.scan(norm(x), h0[i], done)
            else:
                x = x + gru.scan(norm(x), h0[i], done, inj, h_inj[:, :, i])
        return masked_logits(self.pi(x), mask), self.vf(x).squeeze(-1)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def policy_kwargs(run_cfg: Optional[dict]) -> dict:
    """``Policy`` keyword arguments of a run config dict (train.TrainConfig as saved)."""
    c = run_cfg or {}
    return {"hidden": c.get("hidden", 512), "pixel_feat": c.get("pixel_feat", 512),
            "chrono_tau": float(c.get("chrono_tau") or 0.0)}
