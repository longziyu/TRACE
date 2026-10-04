"""Multimodal temporal encoding, mixture generation, and consequence ensembles."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .observations import GROUP_DIMS


class ImageEncoder(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, 32, 5, 2, 2), nn.GELU(),
                                    nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, hidden))

    def forward(self, x):
        return self.layers(x)


class SceneEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden
        self.rgb, self.bev = ImageEncoder(3, d), ImageEncoder(2, d)
        self.groups = nn.ModuleDict({k: nn.Sequential(nn.Linear(n, d), nn.GELU(), nn.Linear(d, d))
                                     for k, n in GROUP_DIMS.items()})
        self.action = nn.Linear(2, d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        self.modality = nn.Parameter(torch.randn(1, 7, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, 8, batch_first=True)
        self.position = nn.Parameter(torch.randn(1, cfg.history, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 8, d * 4, dropout=0.0, batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)

    def forward(self, obs):
        b, length = obs["rgb"].shape[:2]
        tokens = [self.rgb(obs["rgb"].flatten(0, 1)), self.bev(obs["bev"].flatten(0, 1))]
        tokens += [net(obs[key].flatten(0, 1)) for key, net in self.groups.items()]
        tokens += [self.action(obs["previous_action"].flatten(0, 1))]
        tokens = torch.stack(tokens, dim=1) + self.modality
        fused, _ = self.cross(self.query.expand(b * length, -1, -1), tokens, tokens, need_weights=False)
        fused = fused.reshape(b, length, -1) + self.position[:, :length]
        return self.temporal(fused)[:, -1]


def project_sequences(raw, previous, cfg):
    """Enforce velocity and per-step acceleration limits; previous is [B, 2]."""
    limits = raw.new_tensor([cfg.max_v, cfg.max_w])
    increments = raw.new_tensor([cfg.max_a, cfg.max_alpha]) * cfg.dt
    prev = previous[:, None, :].expand(raw.shape[0], raw.shape[1], 2)
    output = []
    for h in range(raw.shape[2]):
        desired = torch.maximum(torch.minimum(raw[:, :, h], limits), -limits)
        desired = torch.stack((desired[..., 0].clamp_min(0), desired[..., 1]), -1)
        step = prev + torch.maximum(torch.minimum(desired - prev, increments), -increments)
        output.append(step)
        prev = step
    return torch.stack(output, dim=2)


class MixtureGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = nn.Linear(cfg.hidden, cfg.learned_candidates * (1 + 4 * cfg.horizon))

    def parameters_for(self, scene):
        c = self.cfg
        out = self.head(scene).reshape(-1, c.learned_candidates, 1 + 4 * c.horizon)
        logits = out[..., 0]
        means = torch.tanh(out[..., 1:1 + 2*c.horizon]).reshape(-1, c.learned_candidates, c.horizon, 2)
        means = means * means.new_tensor([c.max_v, c.max_w])
        scales = out[..., 1+2*c.horizon:].reshape_as(means).clamp(-5, 0).exp()
        return logits, means, scales

    def candidates(self, scene, previous, sample=False):
        _, means, scales = self.parameters_for(scene)
        raw = means + torch.randn_like(means) * scales if sample else means
        c = self.cfg
        # One candidate from every component prevents dropping low-weight modes.
        basics = raw.new_tensor([[0, 0], [c.max_v, 0], [0, c.max_w], [0, -c.max_w]])
        basics = basics[None, :, None, :].expand(len(scene), 4, c.horizon, 2)
        return project_sequences(torch.cat([raw, basics], 1), previous, c)

    def nll(self, scene, sequences):
        logits, mean, scale = self.parameters_for(scene)
        x = sequences[:, :, None]
        logp = (-0.5*((x-mean[:, None])/scale[:, None]).square()
                - scale[:, None].log() - 0.5*math.log(2*math.pi)).sum((-1, -2))
        return -torch.logsumexp(logp + logits.log_softmax(-1)[:, None], dim=-1)


class TraceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = SceneEncoder(cfg)
        self.generator = MixtureGenerator(cfg)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(cfg.hidden+2*cfg.horizon, cfg.hidden),
                                                 nn.GELU(), nn.Linear(cfg.hidden, 4))
                                    for _ in range(cfg.ensemble)])

    def predict(self, scene, candidates):
        shared = scene[:, None].expand(-1, candidates.shape[1], -1)
        x = torch.cat((shared, candidates.flatten(-2)), -1)
        raw = torch.stack([head(x) for head in self.heads], dim=1)
        return torch.cat((raw[..., :1].sigmoid(), F.softplus(raw[..., 1:])), dim=-1)
