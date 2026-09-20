"""Pointer-network policy for MaskablePPO."""

from __future__ import annotations

import torch
import torch.nn as nn
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class TaskAssignmentNet(nn.Module):
    def __init__(
        self,
        task_dim: int,
        sat_dim: int,
        n_sats: int,
        global_dim: int,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.task_dim = task_dim
        self.sat_dim = sat_dim
        self.n_sats = n_sats
        self.global_dim = global_dim
        self.hidden = hidden
        self.task_enc = nn.Sequential(nn.Linear(task_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.sat_enc = nn.Sequential(nn.Linear(sat_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.global_enc = nn.Sequential(nn.Linear(global_dim, hidden), nn.ReLU())
        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.defer_head = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t0 = self.task_dim
        t1 = t0 + self.n_sats * self.sat_dim
        task = obs[:, :t0]
        sats = obs[:, t0:t1].reshape(-1, self.n_sats, self.sat_dim)
        glob = obs[:, t1:]
        return task, sats, glob

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task, sats, glob = self.split(obs)
        h_t = self.task_enc(task)
        h_s = self.sat_enc(sats)
        h_g = self.global_enc(glob)
        q = self.W_q(h_t).unsqueeze(1)
        k = self.W_k(h_s)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        defer = self.defer_head(h_t + h_g)
        logits = torch.cat([scores, defer], dim=-1)
        pooled = h_s.mean(dim=1)
        value = self.value_head(torch.cat([h_t, pooled, h_g], dim=-1))
        return logits, value


class _IdentityExtractor(nn.Module):
    latent_dim_pi = 1
    latent_dim_vf = 1

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = features[..., :1]
        return z, z


class TaskPointerPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_sats: int = 4,
        task_dim: int = 10,
        sat_dim: int = 19,
        global_dim: int = 2,
        hidden: int = 128,
        **kwargs,
    ) -> None:
        self._pointer_cfg = dict(
            n_sats=n_sats,
            task_dim=task_dim,
            sat_dim=sat_dim,
            global_dim=global_dim,
            hidden=hidden,
        )
        kwargs["net_arch"] = []
        kwargs["ortho_init"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _IdentityExtractor()

    def _build(self, lr_schedule) -> None:
        cfg = self._pointer_cfg
        self.net = TaskAssignmentNet(
            task_dim=cfg["task_dim"],
            sat_dim=cfg["sat_dim"],
            n_sats=cfg["n_sats"],
            global_dim=cfg["global_dim"],
            hidden=cfg["hidden"],
        )
        self.action_dist = MaskableCategoricalDistribution(int(self.action_space.n))
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.mlp_extractor = _IdentityExtractor()
        self.optimizer = self.optimizer_class(
            self.net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _logits_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(obs, dict):
            obs = obs[next(iter(obs))]
        logits, value = self.net(obs.float())
        return logits, value

    def _dist(self, logits: torch.Tensor, action_masks=None):
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs, deterministic: bool = False, action_masks=None):
        logits, values = self._logits_value(obs)
        distribution = self._dist(logits, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, values = self._logits_value(obs)
        distribution = self._dist(logits, action_masks)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs, action_masks=None):
        logits, _ = self._logits_value(obs)
        return self._dist(logits, action_masks)

    def predict_values(self, obs):
        _, values = self._logits_value(obs)
        return values

    def _predict(self, observation, deterministic: bool = False, action_masks=None):
        logits, _ = self._logits_value(observation)
        distribution = self._dist(logits, action_masks)
        return distribution.get_actions(deterministic=deterministic)


def make_pointer_policy_kwargs(env, hidden: int = 128) -> dict:
    spec = env.unwrapped.feat_spec
    return dict(
        n_sats=spec.n_sats,
        task_dim=spec.task_dim,
        sat_dim=spec.sat_dim,
        global_dim=spec.global_dim,
        hidden=hidden,
    )
