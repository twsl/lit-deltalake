from collections.abc import Callable
from typing import Any, Literal

from torch.utils.data import BatchSampler, DataLoader, Sampler

from lit_deltalake.datasets.conversion import dataclass_collate, tensordict_collate
from lit_deltalake.datasets.iterable_dataset import DeltaIterableDataset
from lit_deltalake.readers.types import SampleT
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError


class DeltaDataLoader(DataLoader[SampleT]):
    """Load batches from a Delta iterable dataset."""

    def __init__(
        self,
        dataset: DeltaIterableDataset[SampleT],
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Any = None,
        batch_sampler: Any = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[SampleT]], object] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable[[int], None] | None = None,
        multiprocessing_context: str | None = None,
        generator: object = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        batch_format: Literal["default", "tensordict"] = "default",
    ) -> None:
        """Create a DataLoader with Delta-specific batching support.

        Args:
            dataset: Iterable dataset that provides Delta table samples.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle samples using the PyTorch DataLoader API.
            sampler: Optional sampler for selecting samples.
            batch_sampler: Optional sampler that yields batches of indices.
            num_workers: Number of worker processes used to load data.
            collate_fn: Function used to combine samples into a batch.
            pin_memory: Whether to copy batches into pinned memory.
            drop_last: Whether to drop the final incomplete batch.
            timeout: Maximum time in seconds to wait for a worker response.
            worker_init_fn: Optional function called when each worker starts.
            multiprocessing_context: Multiprocessing context used by workers.
            generator: Optional random generator used by the loader.
            prefetch_factor: Number of batches loaded in advance by each worker.
            persistent_workers: Whether to keep workers alive between iterations.
            pin_memory_device: Device used for pinned memory.
            in_order: Whether to yield batches in worker order.
            batch_format: Batch collation format.
        """
        if batch_size is not None and batch_size < 1:
            raise InvalidDatasetConfigurationError("batch_size must be positive.")
        if num_workers < 0:
            raise InvalidDatasetConfigurationError("num_workers must not be negative.")
        if persistent_workers and num_workers == 0:
            raise InvalidDatasetConfigurationError("persistent_workers requires num_workers > 0.")
        if prefetch_factor is not None and num_workers == 0:
            raise InvalidDatasetConfigurationError("prefetch_factor requires num_workers > 0.")
        if num_workers and not dataset.reader.capabilities.supports_worker_processes:
            raise InvalidDatasetConfigurationError(
                f"{type(dataset.reader).__name__} cannot run in DataLoader worker processes; use num_workers=0."
            )
        if batch_format not in {"default", "tensordict"}:
            raise InvalidDatasetConfigurationError("batch_format must be 'default' or 'tensordict'.")

        selected_collate_fn = (
            collate_fn
            if collate_fn is not None
            else (tensordict_collate if batch_format == "tensordict" else dataclass_collate)
        )
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=selected_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            in_order=in_order,
        )
