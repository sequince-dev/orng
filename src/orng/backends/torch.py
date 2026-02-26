from __future__ import annotations

from dataclasses import dataclass
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
        samples = torch._standard_gamma(
            concentration, generator=self._generator
        )
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


@dataclass(frozen=True)
class TorchFunctionalState:
    generator_state: Any
    device: Any


class TorchFunctionalBackend:
    def __init__(self, *, device: Any | None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyTorch backend requires the 'torch' package to be "
                "installed. "
                "Install it with `pip install orng[torch]`."
            ) from exc

        self._torch = torch
        self._device = device if device is not None else torch.device("cpu")

    def init_state(
        self,
        *,
        seed: int | None,
        generator: Any | None,
    ) -> TorchFunctionalState:
        torch = self._torch
        if generator is None:
            gen = torch.Generator(device=self._device)
            if seed is not None:
                gen.manual_seed(seed)
        elif isinstance(generator, torch.Generator):
            gen = generator
        else:
            raise TypeError(
                "generator must be a torch.Generator when using the PyTorch "
                "backend."
            )
        state = gen.get_state().clone()
        gen_device = getattr(gen, "device", self._device)
        return TorchFunctionalState(generator_state=state, device=gen_device)

    def _generator_from_state(self, state: TorchFunctionalState) -> Any:
        gen = self._torch.Generator(device=state.device)
        gen.set_state(state.generator_state.clone())
        return gen

    def _next_state_and_result(
        self,
        gen: Any,
        state: TorchFunctionalState,
        result: Any,
    ) -> tuple[TorchFunctionalState, Any]:
        return (
            TorchFunctionalState(
                generator_state=gen.get_state().clone(),
                device=state.device,
            ),
            result,
        )

    def random(
        self,
        state: TorchFunctionalState,
        *,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[TorchFunctionalState, Any]:
        shape = normalize_shape(size)
        torch = self._torch
        sample_dtype = dtype if dtype is not None else torch.float32
        gen = self._generator_from_state(state)
        if not shape:
            result = torch.rand(
                (1,),
                generator=gen,
                device=state.device,
                dtype=sample_dtype,
            )[0]
        else:
            result = torch.rand(
                shape,
                generator=gen,
                device=state.device,
                dtype=sample_dtype,
            )
        return self._next_state_and_result(gen, state, result)

    def uniform(
        self,
        state: TorchFunctionalState,
        *,
        low: Any,
        high: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[TorchFunctionalState, Any]:
        shape = normalize_shape(size)
        torch = self._torch
        sample_dtype = dtype if dtype is not None else torch.float32
        gen = self._generator_from_state(state)
        sample_shape = shape if shape else (1,)
        base = torch.rand(
            sample_shape,
            generator=gen,
            device=state.device,
            dtype=sample_dtype,
        )
        low_tensor = torch.as_tensor(
            low,
            device=state.device,
            dtype=sample_dtype,
        )
        high_tensor = torch.as_tensor(
            high,
            device=state.device,
            dtype=sample_dtype,
        )
        result = low_tensor + (high_tensor - low_tensor) * base
        if not shape:
            result = result[0]
        return self._next_state_and_result(gen, state, result)

    def normal(
        self,
        state: TorchFunctionalState,
        *,
        loc: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[TorchFunctionalState, Any]:
        shape = normalize_shape(size)
        torch = self._torch
        sample_dtype = dtype if dtype is not None else torch.float32
        gen = self._generator_from_state(state)
        if not shape:
            sample = torch.randn(
                (1,),
                generator=gen,
                device=state.device,
                dtype=sample_dtype,
            )
            result = sample[0] * scale + loc
        else:
            base = torch.randn(
                shape,
                generator=gen,
                device=state.device,
                dtype=sample_dtype,
            )
            result = base * scale + loc
        return self._next_state_and_result(gen, state, result)

    def gamma(
        self,
        state: TorchFunctionalState,
        *,
        shape: Any,
        scale: Any,
        size: SizeLike,
        dtype: Any | None,
    ) -> tuple[TorchFunctionalState, Any]:
        sample_shape = normalize_shape(size)
        torch = self._torch
        sample_dtype = dtype if dtype is not None else torch.float32
        gen = self._generator_from_state(state)
        concentration = torch.as_tensor(
            shape,
            device=state.device,
            dtype=sample_dtype,
        )
        scale_tensor = torch.as_tensor(
            scale,
            device=state.device,
            dtype=sample_dtype,
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
        samples = torch._standard_gamma(concentration, generator=gen)
        result = samples * scale_tensor
        return self._next_state_and_result(gen, state, result)

    def choice(
        self,
        state: TorchFunctionalState,
        population: int | Any,
        *,
        size: SizeLike,
        replace: bool,
        probabilities: Any | None,
    ) -> tuple[TorchFunctionalState, Any]:
        shape = normalize_shape(size)
        num_samples = total_size(shape)
        torch = self._torch
        gen = self._generator_from_state(state)

        if isinstance(population, int):
            pop_size = population
            population_tensor = None
        else:
            population_tensor = torch.as_tensor(
                population,
                device=state.device,
            )
            if population_tensor.ndim == 0:
                population_tensor = population_tensor.reshape(1)
            pop_size = int(population_tensor.shape[0])

        if probabilities is not None:
            probs_tensor = torch.as_tensor(
                probabilities,
                dtype=torch.double,
                device=state.device,
            )
            if probs_tensor.ndim != 1 or probs_tensor.shape[0] != pop_size:
                raise ValueError(
                    "Probabilities must be a 1-D array matching "
                    "the population."
                )
            indices = torch.multinomial(
                probs_tensor,
                num_samples,
                replacement=replace,
                generator=gen,
            )
        else:
            if not replace and num_samples > pop_size:
                raise ValueError(
                    "Cannot take a larger sample than population when "
                    "replace=False."
                )
            if replace:
                indices = torch.randint(
                    pop_size,
                    (num_samples,),
                    generator=gen,
                    device=state.device,
                )
            else:
                indices = torch.randperm(
                    pop_size,
                    generator=gen,
                    device=state.device,
                )[:num_samples]

        if population_tensor is not None:
            draws = population_tensor.index_select(0, indices)
        else:
            draws = indices

        if shape:
            result = draws.reshape(shape)
        else:
            result = draws[0]
        return self._next_state_and_result(gen, state, result)


__all__ = ["TorchBackend", "TorchFunctionalBackend"]
