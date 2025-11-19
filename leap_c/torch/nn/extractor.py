"""This module contains classes for feature extraction from observations.

We provide an abstraction to allow algorithms to be applied to different
types of observations and using different neural network architectures.
"""

from abc import ABC, abstractmethod
from typing import Literal

import gymnasium as gym
import torch.nn as nn

from leap_c.torch.nn.custom import TrajectoryTCN, flatten_and_concat
from leap_c.torch.nn.deepset import DeepSetLayer
from leap_c.torch.nn.scale import min_max_scaling


class Extractor(nn.Module, ABC):
    """An abstract class for feature extraction from observations."""

    def __init__(self, observation_space: gym.Space) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the environment.
        """
        super().__init__()
        self.observation_space = observation_space

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Returns the embedded vector size."""


class ScalingExtractor(Extractor):
    """An extractor that returns the input normalized to the range [0, 1], using min-max scaling."""

    def __init__(self, observation_space: gym.spaces.Box) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the environment. Only works for Box spaces.
        """
        super().__init__(observation_space)

        if len(observation_space.shape) != 1:  # type: ignore
            raise ValueError("ScalingExtractor only supports 1D observations.")

    def forward(self, x):
        """Returns the input normalized to the range [0, 1], using min-max scaling.

        Args:
            x: The input tensor.

        Returns:
            The normalized tensor.
        """
        y = min_max_scaling(x, self.observation_space)  # type: ignore
        return y

    @property
    def output_size(self) -> int:
        return self.observation_space.shape[0]  # type: ignore


class IdentityExtractor(Extractor):
    """An extractor that returns the input as is."""

    def __init__(self, observation_space: gym.Space) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the environment.
        """
        super().__init__(observation_space)
        assert (
                len(observation_space.shape) == 1  # type: ignore
        ), "IdentityExtractor only supports 1D observations."

    def forward(self, x):
        """Returns the input as is.

        Args:
            x: The input tensor.

        Returns:
            The input tensor.
        """
        return x

    @property
    def output_size(self) -> int:
        return self.observation_space.shape[0]  # type: ignore


ExtractorName = Literal["identity", "scaling"]


def get_input_dims(dim_nx: int, dim_nref: int, dim_nfeat: int) -> dict[str, int]:
    dims = {}
    dims['nx'] = dim_nx
    dims['ref'] = 3
    dims['nref'] = dim_nref
    dims['nref_flat'] = dims['ref'] * dim_nref  # 3 positions per step
    dims['cov'] = 3
    dims['feat'] = 3
    dims['nfeat'] = dim_nfeat
    dims['nfeat_flat'] = dims['feat'] * dim_nfeat

    dims['n_total'] = dims['nx'] + dims['nref_flat'] + dims['nfeat_flat'] + dims['cov'] + dims['nfeat']
    dims['idxs_state'] = slice(0, dims['nx'])
    dims['idxs_ref'] = slice(dims['nx'], dims['nx'] + dims['nref_flat'])
    dims['idxs_cov'] = slice(dims['nx'] + dims['nref_flat'], dims['nx'] + dims['nref_flat'] + dims['cov'])
    dims['idxs_feat'] = slice(dims['nx'] + dims['nref_flat'] + dims['cov'], dims['n_total'] - dims['nfeat'])
    dims['idxs_feat_valid'] = slice(dims['n_total'] - dims['nfeat'], dims['n_total'])
    return dims


class QuadrotorExtractor(Extractor):
    def __init__(self, observation_space: gym.spaces.Box, dim_nx: int = 17, dim_nref: int = 51,
                 dim_nfeat: int = 20) -> None:
        """Initializes the extractor.

        Args:
            observation_space: The observation space of the dorne environment.

        """
        super().__init__(observation_space)
        self.dims = get_input_dims(dim_nx, dim_nref, dim_nfeat)
        embedded_dim = 16
        self.trajectory_tcn = TrajectoryTCN(embed_dim=embedded_dim)
        self.deep_set = DeepSetLayer(out_dim=embedded_dim)
        self.output_dim = self.dims['nx'] + self.dims['cov'] + 2*embedded_dim

        # if len(observation_space.shape) != 1:  # type: ignore
        #    raise ValueError("QuadrotorExtractor only supports 1D observations.")

    def forward(self, x):
        """Returns the input normalized to the range [0, 1], using min-max scaling.

        Args:
            x: The input tensor.

        Returns:
            The normalized tensor.
        """
        y_state = x[:, self.dims['idxs_state']]
        x_ref = x[:, self.dims['idxs_ref']].reshape(x.shape[0], self.dims['nref'], self.dims['ref'])
        y_cov = x[:, self.dims['idxs_cov']]
        x_feat = x[:, self.dims['idxs_feat']].reshape(x.shape[0], self.dims['nfeat'], self.dims['feat'])
        x_feat_valid = x[:, self.dims['idxs_feat_valid']]

        x_ref_rel = x_ref - y_state[:, :3].unsqueeze(1)  # relative to ego position
        y_traj = self.trajectory_tcn(x_ref_rel)

        # compute quaternion difference of features to ego quaternion
        x_pos = y_state[:, :3]
        x_quat = y_state[:, 3:7]
        x_vec_ego2feat = x_feat - x_pos.unsqueeze(1)  # vector from ego to features
        x_feat_diff = quaternion_difference_ego_to_vectors(x_quat, x_vec_ego2feat)
        y_feat_emb = self.deep_set(x_feat_diff, x_feat_valid)

        y = flatten_and_concat(y_state, y_cov, y_traj, y_feat_emb)
        # y = min_max_scaling(x, self.observation_space)  # type: ignore
        return y

    @property
    def output_size(self) -> int:
        return self.output_dim # type: ignore


EXTRACTOR_REGISTRY = {
    "identity": IdentityExtractor,
    "scaling": ScalingExtractor,
    "drone": QuadrotorExtractor,
}


def get_extractor_cls(name: ExtractorName):
    try:
        return EXTRACTOR_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown extractor type: {name}")


import torch


def quat_mul(q1, q2):
    """
    Hamilton product of two quaternions.
    q1, q2: (..., 4) [w, x, y, z]
    returns: (..., 4)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def direction_vectors_to_quat(v, eps=1e-8):
    """
    Convert direction vectors to quaternions that rotate +Z to v.

    v: (B, N, 3) direction vectors (not necessarily normalized)
    returns: (B, N, 4) quaternions [w, x, y, z]
    """
    # Normalize input vectors
    v_norm = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

    B, N, _ = v_norm.shape
    device = v.device
    dtype = v.dtype

    # Reference axis: +Z
    ref = torch.tensor([0., 0., 1.], device=device, dtype=dtype)
    ref = ref.view(1, 1, 3).expand(B, N, 3)  # (B, N, 3)

    # Dot and cross between ref and v
    dot = (ref * v_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (B, N, 1)
    axis = torch.cross(ref, v_norm, dim=-1)  # (B, N, 3)

    angle = torch.acos(dot)  # (B, N, 1)

    # General case
    axis_norm = axis.norm(dim=-1, keepdim=True)
    safe_axis = axis / axis_norm.clamp_min(eps)

    half_angle = 0.5 * angle
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)

    q = torch.cat([cos_half, safe_axis * sin_half], dim=-1)  # (B, N, 4)

    # Handle near-parallel and anti-parallel cases more stably
    parallel_thresh = 1.0 - 1e-6
    anti_parallel_thresh = -1.0 + 1e-6

    # v ~= +Z  -> identity rotation
    mask_parallel = dot.squeeze(-1) > parallel_thresh
    if mask_parallel.any():
        q[mask_parallel] = torch.tensor(
            [1., 0., 0., 0.], device=device, dtype=dtype
        )

    # v ~= -Z -> 180deg rotation around any axis orthogonal to Z, choose X
    mask_antiparallel = dot.squeeze(-1) < anti_parallel_thresh
    if mask_antiparallel.any():
        # 180deg about X: [0, 1, 0, 0]
        q[mask_antiparallel] = torch.tensor(
            [0., 1., 0., 0.], device=device, dtype=dtype
        )

    return q


def quaternion_difference_ego_to_vectors(ego_q, orientation_vectors, eps=1e-8):
    """
    Compute quaternion differences between an ego orientation quaternion and
    N 3D orientation vectors.

    ego_q: (B, 4)   quaternions [w, x, y, z]
    orientation_vectors: (B, N, 3) direction vectors

    returns:
        q_diff: (B, N, 4) quaternions [w, x, y, z]
                rotation that takes ego orientation to the orientation of each vector.
    """
    # Ensure unit quaternions for ego
    ego_q = ego_q / ego_q.norm(dim=-1, keepdim=True).clamp_min(eps)  # (B, 4)

    B, N, _ = orientation_vectors.shape

    # Convert orientation vectors to quaternions (world orientation)
    q_vec = direction_vectors_to_quat(orientation_vectors, eps=eps)  # (B, N, 4)

    # Inverse (conjugate) of ego quaternion (assumed unit)
    ego_q_inv = ego_q.clone()
    ego_q_inv[..., 1:] = -ego_q_inv[..., 1:]  # (B, 4)
    ego_q_inv = ego_q_inv.unsqueeze(1).expand(B, N, 4)  # (B, N, 4)

    # Difference: q_diff = q_vec ⊗ ego_q^{-1}
    q_diff = quat_mul(q_vec, ego_q_inv)  # (B, N, 4)

    # Optionally re-normalize
    q_diff = q_diff / q_diff.norm(dim=-1, keepdim=True).clamp_min(eps)

    return q_diff
