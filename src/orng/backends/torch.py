from __future__ import annotations

from typing import Any

from .._utils import SizeLike, normalize_shape, total_size


class TorchBackend:
    def __init__(
        self,
        *,
        seed: int | None,
        generator: Any | None,
        device: Any | None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyTorch backend requires the 'torch' package to be installed. "
                "Install it with `pip install orng[torch]`."
            ) from exc

        self._torch = torch
        self._device = device if device is not None else torch.device("cpu")
        if generator is None:
            self._generator = torch.Generator(device=self._device)
            if seed is not None:
                self._generator.manual_seed(seed)
        elif isinstance(generator, torch.Generator):
            self._generator = generator
        else:
            raise TypeError(
                "generator must be a torch.Generator when using the PyTorch "
                "backend."
            )

    def random(self, *, size: SizeLike, dtype: Any | None) -> Any:
        shape = normalize_shape(size)
        dtype = dtype if dtype is not None else self._torch.float32
        if not shape:
            result = self._torch.rand(
                (1,),
                generator=self._generator,
                device=self._device,
                dtype=dtype,
            )
            return result[0]
        return self._torch.rand(
            shape,
            generator=self._generator,
            device=self._device,
            dtype=dtype,
        )

    def uniform(
        self,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        shape = normalize_shape(size)
        dtype = dtype if dtype is not None else self._torch.float32
        sample_shape = shape if shape else (1,)
        base = self._torch.rand(
            sample_shape,
            generator=self._generator,
            device=self._device,
            dtype=dtype,
        )
        low_tensor = self._torch.as_tensor(
            low, device=self._device, dtype=dtype
        )
        high_tensor = self._torch.as_tensor(
            high, device=self._device, dtype=dtype
        )
        result = low_tensor + (high_tensor - low_tensor) * base
        if not shape:
            return result[0]
        return result

    def normal(
        self,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        shape = normalize_shape(size)
        dtype = dtype if dtype is not None else self._torch.float32
        if not shape:
            sample = self._torch.randn(
                (1,),
                generator=self._generator,
                device=self._device,
                dtype=dtype,
            )
            return sample[0] * scale + loc
        base = self._torch.randn(
            shape,
            generator=self._generator,
            device=self._device,
            dtype=dtype,
        )
        return base * scale + loc

    def gamma(
        self,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> Any:
        sample_shape = normalize_shape(size)
        torch = self._torch
        dtype = dtype if dtype is not None else torch.float32
        concentration = torch.as_tensor(
            shape,
            device=self._device,
            dtype=dtype,
        )
        scale_tensor = torch.as_tensor(
            scale,
            device=self._device,
            dtype=dtype,
        )
        concentration, scale_tensor = torch.broadcast_tensors(
            concentration, scale_tensor
        )
        if torch.any(concentration <= 0):
            raise ValueError("shape parameters for gamma must be positive.")
        if sample_shape:
            prefix = (None,) * len(sample_shape)
            concentration = concentration[prefix].expand(
                sample_shape + concentration.shape
            )
            scale_tensor = scale_tensor[prefix].expand(
                sample_shape + scale_tensor.shape
            )
        samples = self._gamma_standard(concentration)
        scaled = samples * scale_tensor
        return scaled

    def choice(
        self,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> Any:
        shape = normalize_shape(size)
        num_samples = total_size(shape)

        if isinstance(population, int):
            pop_size = population
            population_tensor = None
        else:
            population_tensor = self._torch.as_tensor(
                population, device=self._device
            )
            if population_tensor.ndim == 0:
                population_tensor = population_tensor.reshape(1)
            pop_size = int(population_tensor.shape[0])

        if probabilities is not None:
            probs_tensor = self._torch.as_tensor(
                probabilities,
                dtype=self._torch.double,
                device=self._device,
            )
            if probs_tensor.ndim != 1 or probs_tensor.shape[0] != pop_size:
                raise ValueError(
                    "Probabilities must be a 1-D array matching the population."
                )
            indices = self._torch.multinomial(
                probs_tensor,
                num_samples,
                replacement=replace,
                generator=self._generator,
            )
        else:
            if not replace and num_samples > pop_size:
                raise ValueError(
                    "Cannot take a larger sample than population when "
                    "replace=False."
                )
            if replace:
                indices = self._torch.randint(
                    pop_size,
                    (num_samples,),
                    generator=self._generator,
                    device=self._device,
                )
            else:
                indices = self._torch.randperm(
                    pop_size,
                    generator=self._generator,
                    device=self._device,
                )[:num_samples]

        if population_tensor is not None:
            draws = population_tensor.index_select(0, indices)
        else:
            draws = indices

        if not shape:
            return draws[0]
        return draws.reshape(shape)

    # Internal helpers ---------------------------------------------------------

    def _gamma_standard(self, concentration_tensor):
        torch = self._torch
        flat_concentration = concentration_tensor.reshape(-1)
        result = torch.empty_like(flat_concentration)
        mask_ge_one = flat_concentration >= 1
        mask_lt_one = ~mask_ge_one

        if torch.any(mask_ge_one):
            result[mask_ge_one] = self._gamma_shape_ge_one(
                flat_concentration[mask_ge_one]
            )
        if torch.any(mask_lt_one):
            conc = flat_concentration[mask_lt_one]
            adjusted = conc + 1.0
            base = self._gamma_shape_ge_one(adjusted)
            u = torch.rand(
                conc.shape,
                generator=self._generator,
                device=self._device,
                dtype=flat_concentration.dtype,
            )
            result[mask_lt_one] = base * u.pow(1.0 / conc)

        return result.reshape(concentration_tensor.shape)

    def _gamma_shape_ge_one(self, concentration):
        torch = self._torch
        dtype = concentration.dtype
        result = torch.empty_like(concentration)
        d = concentration - 1.0 / 3.0
        c = 1.0 / torch.sqrt(9.0 * d)
        done = torch.zeros_like(concentration, dtype=torch.bool)

        while True:
            pending = ~done
            if not torch.any(pending):
                break
            pending_size = int(pending.sum().item())
            x = torch.randn(
                (pending_size,),
                generator=self._generator,
                device=self._device,
                dtype=dtype,
            )
            d_pending = d[pending]
            c_pending = c[pending]
            v = (1.0 + c_pending * x) ** 3
            u = torch.rand(
                (pending_size,),
                generator=self._generator,
                device=self._device,
                dtype=dtype,
            )
            positive = v > 0
            log_v = torch.zeros_like(v)
            if torch.any(positive):
                log_v[positive] = torch.log(v[positive])
            cond = (u < 1 - 0.331 * (x**4)) | (
                torch.log(u) < 0.5 * x**2 + d_pending * (1 - v + log_v)
            )
            accept = positive & cond
            if not torch.any(accept):
                continue
            pending_indices = pending.nonzero(as_tuple=False).squeeze(-1)
            selected = pending_indices[accept]
            result[selected] = d[selected] * v[accept]
            done[selected] = True

        return result


__all__ = ["TorchBackend"]
