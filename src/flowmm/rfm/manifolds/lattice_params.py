"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import torch
from geoopt import Euclidean
from geoopt.utils import broadcast_shapes, size2shape
from omegaconf import OmegaConf
from typing_extensions import Self
from flowmm.cfg_utils import dataset_options, init_loaders


class NoDtypeDevice:
    @property
    def dtype(self) -> None:
        """the manifold requires bool tensors, this disrupts introspection default"""
        return None

    @property
    def device(self) -> None:
        """gets mapped to the right device as necessary, this disrupts introspection default"""
        return None


class PositiveEuclidean(NoDtypeDevice, Euclidean):
    name = "PositiveEuclidean"
    reversible = False
    ndim = 1

    def __init__(
        self, loc: torch.Tensor = torch.zeros(3), scale: torch.Tensor = torch.ones(3)
    ):
        super().__init__(ndim=1)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @staticmethod
    def clamp(x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=torch.zeros_like(x))

    def _get_log_normal(self, dtype, device) -> torch.distributions.Distribution:
        return torch.distributions.LogNormal(
            self.loc.to(dtype=dtype, device=device),
            self.scale.to(dtype=dtype, device=device),
            validate_args=False,
        )

    def random_lognormal(
        self, *size, dtype: torch.dtype = None, device: torch.device | str = None
    ) -> torch.Tensor:
        assert (
            size[-1] == self.loc.shape[-1]
        ), f"requested {size=} did not match last dim of {self.loc=}"
        assert (
            size[-1] == self.scale.shape[-1]
        ), f"requested {size=} did not match last dim of {self.scale=}"
        d = self._get_log_normal(dtype, device)
        return d.sample(size[:-1])

    def random_base(self, *args, **kwargs):
        return self.random_lognormal(*args, **kwargs)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        d = self._get_log_normal(dtype=x.dtype, device=x.device)
        return d.log_prob(x)

    def extra_repr(self):
        return f"loc={self.loc}, scale={self.scale}"


class UnconstrainedCompact(NoDtypeDevice, Euclidean):
    name = "UnconstrainedCompact"
    ndim = 1

    def __init__(self, low: float | torch.Tensor, high: float | torch.Tensor):
        """generates in the unconstrained space since the flow lives there"""
        super().__init__(ndim=1)
        self.register_buffer("low", torch.Tensor(low))
        self.register_buffer("high", torch.Tensor(high))

    def _get_uniform(
        self, dtype: torch.dtype, device: torch.device | str
    ) -> torch.distributions.Distribution:
        return torch.distributions.Uniform(
            low=self.low.to(dtype=dtype, device=device),
            high=self.high.to(dtype=dtype, device=device),
            validate_args=False,
        )

    def get_unconstrained_to_constrained(
        self, dtype: torch.dtype, device: torch.device | str
    ) -> torch.distributions.Transform:
        return torch.distributions.biject_to(self._get_uniform(dtype, device).support)

    def get_constrained_to_unconstrained(
        self, dtype: torch.dtype, device: torch.device | str
    ) -> torch.distributions.Transform:
        return self.get_unconstrained_to_constrained(dtype, device).inv

    def get_unconstrained_dist(
        self, dtype: torch.dtype, device: torch.device | str
    ) -> torch.distributions.TransformedDistribution:
        return torch.distributions.TransformedDistribution(
            self._get_uniform(dtype, device),
            self.get_constrained_to_unconstrained(dtype, device),
        )

    def random_unconstrained(
        self, *size, dtype: torch.dtype = None, device: torch.device | str = None
    ) -> torch.Tensor:
        assert (
            size[-1] == self.low.shape[-1]
        ), f"requested {size=} did not match last dim of {self.low=}"
        assert (
            size[-1] == self.high.shape[-1]
        ), f"requested {size=} did not match last dim of {self.high=}"
        d = self.get_unconstrained_dist(dtype, device)
        return d.sample(size[:-1])

    def random_base(self, *args, **kwargs):
        return self.random_unconstrained(*args, **kwargs)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def extra_repr(self):
        return f"low={self.low}, high={self.high}"


class LatticeParams(NoDtypeDevice, Euclidean):
    name = "LatticeParams"
    reversible = True
    ndim = 1

    def __init__(
        self,
        length_log_loc: torch.Tensor = torch.zeros(3),
        length_log_scale: torch.Tensor = torch.ones(3),
        length_inner_coef: float | None = None,
    ):
        super().__init__(ndim=1)
        self.positive_euclidean = PositiveEuclidean(length_log_loc, length_log_scale)
        self.bounds_deg = (59.9, 120.1)
        t = torch.Tensor(self.bounds_deg)
        self.unconstrained_compact = UnconstrainedCompact(
            t[0].expand(3), t[1].expand(3)
        )
        if length_inner_coef is None:
            self.length_inner_coef = 1.0
        else:
            self.length_inner_coef = length_inner_coef

    @classmethod
    def from_dataset(
        cls, dataset: dataset_options, length_inner_coef: float | None = None
    ) -> Self:
        lattice_params_stats = get_lattice_params_stats(dataset)
        return cls(
            lattice_params_stats.length_log_mean,
            lattice_params_stats.length_log_std,
            length_inner_coef,
        )

    @staticmethod
    def dim(dim_coords: int) -> int:
        return dim_coords * 2

    @staticmethod
    def cat(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return torch.cat([lengths, angles], dim=-1)

    @staticmethod
    def _dim_if_divisible_by_2(x: torch.Tensor) -> int:
        dim = x.shape[-1]
        assert (
            dim % 2 == 0
        ), f"{x=}'s last dimension was not divisible by two and therefore not lengths and angles"
        return dim

    @classmethod
    def split(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dim = cls._dim_if_divisible_by_2(x)
        return x[..., : dim // 2], x[..., dim // 2 :]

    @classmethod
    def deg2rad(cls, x: torch.Tensor) -> torch.Tensor:
        lengths, angles_deg = cls.split(x)
        angles_rad = torch.deg2rad(angles_deg)
        return cls.cat(lengths, angles_rad)

    @classmethod
    def rad2deg(cls, x: torch.Tensor) -> torch.Tensor:
        lengths, angles_rad = cls.split(x)
        angles_deg = torch.rad2deg(angles_rad)
        return cls.cat(lengths, angles_deg)

    def deg2uncontrained(self, x: torch.Tensor) -> torch.Tensor:
        lengths, angles = self.split(x)
        t = self.unconstrained_compact.get_constrained_to_unconstrained(
            x.dtype, x.device
        )
        angles_uncon = t(angles)
        return self.cat(lengths, angles_uncon)

    def uncontrained2deg(self, x: torch.Tensor) -> torch.Tensor:
        lengths, angles = self.split(x)
        t = self.unconstrained_compact.get_unconstrained_to_constrained(
            x.dtype, x.device
        )
        angles_uncon = t(angles)
        return self.cat(lengths, angles_uncon)

    # @classmethod
    # def clamp(cls, x: torch.Tensor) -> torch.Tensor:
    #     lengths, angles = cls.split(x)
    #     lengths = lengths.clamp(min=torch.zeros_like(lengths))
    #     return cls.cat(lengths, angles)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v

        # here is where we weigh the length vs angle
        l, a = self.split(inner)
        l = l * self.length_inner_coef
        inner = self.cat(l, a)

        inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        x_shape = x.shape[: -self.ndim] + (1,) * self.ndim * keepdim

        i_shape = inner.shape
        target_shape = broadcast_shapes(x_shape, i_shape)
        return inner.expand(target_shape)

    def random_base(
        self, *size, dtype: torch.dtype = None, device: torch.device | str = None
    ) -> torch.Tensor:
        lengths = self.positive_euclidean.random(
            *(*size[:-1], size[-1] // 2), dtype=dtype, device=device
        )
        angles = self.unconstrained_compact.random(
            *(*size[:-1], size[-1] // 2), dtype=lengths.dtype, device=lengths.device
        )
        return self.cat(lengths, angles)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
        lengths, angles_rad = self.split(x)
        length_log_prob = self.positive_euclidean.base_logprob(lengths)
        angles_rad_log_prob = FlatTorus.base_logprob(angles_rad)
        return length_log_prob + angles_rad_log_prob

    def extra_repr(self):
        return f"length_log_loc={self.loc}, length_log_scale={self.scale}, length_inner_coef={self.length_inner_coef}"

    def abits_clamp(self, x: torch.Tensor) -> torch.Tensor:
        return x

class LatticeParamsEuclidean(NoDtypeDevice, Euclidean):
    name = "LatticeParamsEuclidean"
    reversible = True
    ndim = 1

    def __init__(
        self,
        dataset: dataset_options,
        epsilon: float = 1e-6,  # Small positive noise
    ):
        """
        Manifold class where lengths and angles are loaded from dataset statistics,
        and sampling is done directly from the dataset stats with optional noise.

        Args:
            dataset (dataset_options): Dataset from which statistics are loaded.
            epsilon (float): Small noise added to sampled data to avoid exact constants.
        """
        super().__init__(ndim=1)
        stats = get_lattice_params_stats(dataset)
        self.register_buffer("length_mean", stats.length_mean)
        self.register_buffer("length_std", stats.length_std + epsilon)
        self.register_buffer("angle_mean", stats.angle_mean)
        self.register_buffer("angle_std", stats.angle_std + epsilon)

    @classmethod
    def from_dataset(cls, dataset: dataset_options, epsilon: float = 1e-6) -> Self:
        """Alternative constructor for easier initialization."""
        return cls(dataset=dataset, epsilon=epsilon)

    @staticmethod
    def dim(dim_coords: int) -> int:
        """Returns the dimensionality of the manifold."""
        return dim_coords * 2

    @staticmethod
    def cat(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Concatenates lengths and angles along the last dimension."""
        return torch.cat([lengths, angles], dim=-1)

    @staticmethod
    def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the concatenated tensor into lengths and angles."""
        dim = x.shape[-1]
        assert dim % 2 == 0, f"Tensor shape must be divisible by 2: got {dim=}"
        return x[..., : dim // 2], x[..., dim // 2 :]

    def random_base(
        self, *size, dtype: torch.dtype = None, device: torch.device | str = None
    ) -> torch.Tensor:
        """
        Samples random lengths and angles directly from dataset statistics.

        Args:
            size (tuple): Shape of the sampled tensor.
            dtype (torch.dtype): Data type of the sampled tensor.
            device (torch.device | str): Device for the sampled tensor.

        Returns:
            torch.Tensor: Sampled tensor of concatenated lengths and angles.
        """
        lengths = torch.normal(
            mean=self.length_mean.to(dtype=dtype, device=device),
            std=self.length_std.to(dtype=dtype, device=device),
            size=(*size[:-1], size[-1] // 2),
        )
        angles = torch.normal(
            mean=self.angle_mean.to(dtype=dtype, device=device),
            std=self.angle_std.to(dtype=dtype, device=device),
            size=(*size[:-1], size[-1] // 2),
        )
        return self.cat(lengths, angles)

    random = random_base

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        """
        Computes the inner product on the manifold.

        Args:
            x (torch.Tensor): Base point on the manifold (not used in Euclidean).
            u (torch.Tensor): Tangent vector.
            v (torch.Tensor, optional): Second tangent vector. Defaults to None.
            keepdim (bool, optional): Whether to keep the last dimension.

        Returns:
            torch.Tensor: Inner product result.
        """
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v

        # Treat lengths and angles differently if needed in the future.
        lengths, angles = self.split(inner)
        inner = self.cat(lengths, angles)

        inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        x_shape = x.shape[: -self.ndim] + (1,) * self.ndim * keepdim
        target_shape = broadcast_shapes(x_shape, inner.shape)
        return inner.expand(target_shape)

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probability of the base distribution.

        Args:
            x (torch.Tensor): Points on the manifold.

        Returns:
            torch.Tensor: Log-probabilities.
        """
        lengths, angles = self.split(x)
        length_log_prob = torch.distributions.Normal(
            self.length_mean.to(dtype=x.dtype, device=x.device),
            self.length_std.to(dtype=x.dtype, device=x.device),
        ).log_prob(lengths).sum(dim=-1)
        angle_log_prob = torch.distributions.Normal(
            self.angle_mean.to(dtype=x.dtype, device=x.device),
            self.angle_std.to(dtype=x.dtype, device=x.device),
        ).log_prob(angles).sum(dim=-1)
        return length_log_prob + angle_log_prob

    def extra_repr(self) -> str:
        """Provides additional information for `repr()`."""
        return (
            f"length_mean={self.length_mean}, length_std={self.length_std}, "
            f"angle_mean={self.angle_mean}, angle_std={self.angle_std}"
        )

class LengthAnglePointManifold(NoDtypeDevice, Euclidean):
    """
    A manifold representing fixed values for both lengths and angles.
    This avoids reliance on dataset statistics and works with fixed values.
    """
    name = "LengthAnglePointManifold"
    reversible = False

    def __init__(self, lengths: torch.Tensor, angles: torch.Tensor, ndim: int = 1):
        """
        Initialize the manifold with fixed lengths and angles.

        Args:
            lengths (torch.Tensor): Fixed lengths for the manifold.
            angles (torch.Tensor): Fixed angles for the manifold.
            ndim (int): Number of dimensions for the manifold.
        """
        super().__init__(ndim=ndim)
        self.register_buffer("lengths", lengths)
        self.register_buffer("angles", angles)

    @staticmethod
    def dim(dim_coords: int) -> int:
        return dim_coords * 2

    @staticmethod
    def cat(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return torch.cat([lengths, angles], dim=-1)

    @staticmethod
    def _dim_if_divisible_by_2(x: torch.Tensor) -> int:
        dim = x.shape[-1]
        assert (
            dim % 2 == 0
        ), f"{x=}'s last dimension was not divisible by two and therefore not lengths and angles"
        return dim

    @classmethod
    def split(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dim = cls._dim_if_divisible_by_2(x)
        return x[..., : dim // 2], x[..., dim // 2 :]
        
    def _combine_lengths_angles(self) -> torch.Tensor:
        """
        Combine lengths and angles into a single tensor for sampling.
        """
        return torch.cat([self.lengths, self.angles], dim=-1)

    def random(self, *size, dtype: torch.dtype = None, device: torch.device | str = None) -> torch.Tensor:
        """
        Sample points from the fixed lengths and angles with a small positive noise.
        """
        combined = self._combine_lengths_angles()
        self._assert_check_shape(size2shape(*size), "x")
        zeros = torch.zeros(*size, device=device, dtype=dtype)

        # Add a small positive noise to avoid degenerate behavior
        noise = torch.randn_like(zeros) * 1e-6

        return combined.to(dtype=dtype, device=device) + zeros + noise

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        """
        Computes the inner product on the manifold.

        Args:
            x (torch.Tensor): Base point on the manifold (not used in Euclidean).
            u (torch.Tensor): Tangent vector.
            v (torch.Tensor, optional): Second tangent vector. Defaults to None.
            keepdim (bool, optional): Whether to keep the last dimension.

        Returns:
            torch.Tensor: Inner product result.
        """
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v

        # Treat lengths and angles differently if needed in the future.
        lengths, angles = self.split(inner)
        inner = self.cat(lengths, angles)

        inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        x_shape = x.shape[: -self.ndim] + (1,) * self.ndim * keepdim
        target_shape = broadcast_shapes(x_shape, inner.shape)
        return inner.expand(target_shape)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x to the fixed lengths and angles.
        """
        combined = self._combine_lengths_angles()
        target_shape = broadcast_shapes(combined.shape, x.shape)
        return combined.expand(target_shape)

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability density for the fixed-point manifold.
        Always returns zero since it's a fixed-point distribution.
        """
        return torch.full_like(x[..., 0], 0.0)

    @staticmethod
    def split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Splits the concatenated tensor into lengths and angles."""
        dim = x.shape[-1]
        assert dim % 2 == 0, f"Tensor shape must be divisible by 2: got {dim=}"
        return x[..., : dim // 2], x[..., dim // 2 :]

    def extra_repr(self):
        return f"lengths={self.lengths}, angles={self.angles}, ndim={self.ndim}"


class NullLatticeParams(NoDtypeDevice, Euclidean):
    name = "LatticeParams"
    reversible = True
    ndim = 1

    def __init__(
        self,
        length_log_loc: torch.Tensor = torch.zeros(3),
        length_log_scale: torch.Tensor = torch.ones(3),
        length_inner_coef: float | None = None,
    ):
        super().__init__(ndim=1)
        # self.positive_euclidean = PositiveEuclidean(length_log_loc, length_log_scale)
        # self.bounds_deg = (59.9, 120.1)
        # t = torch.Tensor(self.bounds_deg)
        # self.unconstrained_compact = UnconstrainedCompact(
            # t[0].expand(3), t[1].expand(3)
        # )
        if length_inner_coef is None:
            self.length_inner_coef = 1.0
        else:
            self.length_inner_coef = length_inner_coef

    @classmethod
    def from_dataset(
        cls, dataset: dataset_options, length_inner_coef: float | None = None
    ) -> Self:
        lattice_params_stats = get_lattice_params_stats(dataset)
        return cls(
            lattice_params_stats.length_log_mean,
            lattice_params_stats.length_log_std,
            length_inner_coef,
        )

    @staticmethod
    def dim(dim_coords: int) -> int:
        return dim_coords * 2

    @staticmethod
    def cat(lengths: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return torch.cat([lengths, angles], dim=-1)

    @staticmethod
    def _dim_if_divisible_by_2(x: torch.Tensor) -> int:
        dim = x.shape[-1]
        assert (
            dim % 2 == 0
        ), f"{x=}'s last dimension was not divisible by two and therefore not lengths and angles"
        return dim

    @classmethod
    def split(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dim = cls._dim_if_divisible_by_2(x)
        return x[..., : dim // 2], x[..., dim // 2 :]

    @classmethod
    def deg2rad(cls, x: torch.Tensor) -> torch.Tensor:
        lengths, angles_deg = cls.split(x)
        angles_rad = torch.deg2rad(angles_deg)
        return cls.cat(lengths, angles_rad)

    @classmethod
    def rad2deg(cls, x: torch.Tensor) -> torch.Tensor:
        lengths, angles_rad = cls.split(x)
        angles_deg = torch.rad2deg(angles_rad)
        return cls.cat(lengths, angles_deg)

    def deg2uncontrained(self, x: torch.Tensor) -> torch.Tensor:
        lengths, angles = self.split(x)
        t = self.unconstrained_compact.get_constrained_to_unconstrained(
            x.dtype, x.device
        )
        angles_uncon = t(angles)
        return self.cat(lengths, angles_uncon)

    def uncontrained2deg(self, x: torch.Tensor) -> torch.Tensor:
        lengths, angles = self.split(x)
        t = self.unconstrained_compact.get_unconstrained_to_constrained(
            x.dtype, x.device
        )
        angles_uncon = t(angles)
        return self.cat(lengths, angles_uncon)

    # @classmethod
    # def clamp(cls, x: torch.Tensor) -> torch.Tensor:
    #     lengths, angles = cls.split(x)
    #     lengths = lengths.clamp(min=torch.zeros_like(lengths))
    #     return cls.cat(lengths, angles)

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v

        # here is where we weigh the length vs angle
        l, a = self.split(inner)
        l = l * self.length_inner_coef
        inner = self.cat(l, a)

        inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        x_shape = x.shape[: -self.ndim] + (1,) * self.ndim * keepdim

        i_shape = inner.shape
        target_shape = broadcast_shapes(x_shape, i_shape)
        return inner.expand(target_shape)

    def random_base(
        self, *size, dtype: torch.dtype = None, device: torch.device | str = None
    ) -> torch.Tensor:
        lengths = self.positive_euclidean.random(
            *(*size[:-1], size[-1] // 2), dtype=dtype, device=device
        )
        angles = self.unconstrained_compact.random(
            *(*size[:-1], size[-1] // 2), dtype=lengths.dtype, device=lengths.device
        )
        return self.cat(lengths, angles)

    random = random_base

    def base_logprob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
        lengths, angles_rad = self.split(x)
        length_log_prob = self.positive_euclidean.base_logprob(lengths)
        angles_rad_log_prob = FlatTorus.base_logprob(angles_rad)
        return length_log_prob + angles_rad_log_prob

    def extra_repr(self):
        return f"length_log_loc={self.loc}, length_log_scale={self.scale}, length_inner_coef={self.length_inner_coef}"

    def abits_clamp(self, x: torch.Tensor) -> torch.Tensor:
        return x



class LatticeParamsNormalBase(LatticeParams):
    name = "LatticeParamsNormalBase"

    def __init__(self, length_inner_coef: float | None = None):
        super().__init__(length_inner_coef=length_inner_coef)
        del self.positive_euclidean
        del self.bounds_deg
        del self.unconstrained_compact

    def random_base(
        self, *size, dtype: torch.dtype = None, device: torch.device | str = None
    ) -> torch.Tensor:
        return torch.randn(*size, dtype=dtype, device=device)

    random = random_base

    def extra_repr(self):
        return f"length_inner_coef={self.length_inner_coef}"


@dataclass
class LatticeParamsStats:
    length_mean: torch.Tensor
    length_std: torch.Tensor
    length_log_mean: torch.Tensor
    length_log_std: torch.Tensor
    angle_uncon_mean: torch.Tensor
    angle_uncon_std: torch.Tensor

    @property
    def mean(self):
        return torch.cat([self.length_mean, self.angle_uncon_mean], dim=-1)

    @property
    def std(self):
        return torch.cat([self.length_std, self.angle_uncon_std], dim=-1)

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def destandardize(self, x_standardized: torch.Tensor) -> torch.Tensor:
        return x_standardized * self.std + self.mean


@cache
def get_lattice_params_stats(
    dataset: dataset_options,
    path: str | Path | None = None,
) -> LatticeParamsStats:
    if path is None:
        file = Path(__file__).parent / f"lattice_params_stats.yaml"
    else:
        file = Path(path)
    file = file.resolve()

    if file.exists():
        stats = OmegaConf.load(str(file))
    else:
        raise FileNotFoundError(
            f"{file=} does not exist, have you computed the mean, std, log_mean, and log_std already?"
        )
    return LatticeParamsStats(**{k: torch.tensor(v) for k, v in stats[dataset].items()})


def compute_lattice_params_stats(
    dataset: dataset_options,
) -> LatticeParamsStats:
    train_loader, *_ = init_loaders(dataset=dataset)
    lengths, angles = [], []
    for batch in train_loader:
        lengths.append(batch.lengths)
        angles.append(batch.angles)
    lengths = torch.cat(lengths, dim=0)
    angles = torch.cat(angles, dim=0)
    lp = LatticeParams()
    const_to_unconst = lp.unconstrained_compact.get_constrained_to_unconstrained(
        angles.dtype, angles.device
    )
    return LatticeParamsStats(
        length_mean=lengths.mean(0),
        length_std=lengths.std(0),
        length_log_mean=lengths.log().mean(0),
        length_log_std=lengths.log().std(0),
        angle_uncon_mean=const_to_unconst(angles).mean(0),
        angle_uncon_std=const_to_unconst(angles).std(0),
    )


if __name__ == "__main__":
    import yaml
    from tqdm import tqdm

    compute_stats = True
    file = Path(__file__).parent / "lattice_params_stats.yaml"
    file = file.resolve()
    if compute_stats:
        print("calculate the stats of p(L) for each dataset")
        stats = {}
        pbar = tqdm(list(dataset_options.__args__))
        for dataset in pbar:
            pbar.set_description(f"{dataset=}")
            lattice_params_stats = compute_lattice_params_stats(dataset)
            stats[dataset] = {
                k: v.cpu().tolist() for k, v in asdict(lattice_params_stats).items()
            }

        with open(file, "w") as f:
            yaml.dump(stats, f)
    else:
        stats = OmegaConf.load(str(file))
