from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import lightning as lightning
from torch.utils.data import DataLoader

from lit_deltalake.dataloaders import DeltaDataLoader
from lit_deltalake.datasets.iterable_dataset import DeltaIterableDataset
from lit_deltalake.readers.types import SampleT
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError

type DatasetFactory[SampleT] = DeltaIterableDataset[SampleT] | Callable[[], DeltaIterableDataset[SampleT]]


@dataclass(frozen=True, slots=True)
class DeltaDataLoaderConfig:
    """DataLoader settings for a data module stage."""

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    multiprocessing_context: str | None = None
    drop_last: bool = False
    batch_format: Literal["default", "tensordict"] = "default"


class DeltaDataModule[SampleT](lightning.LightningDataModule):
    """Create stage-specific Delta datasets and their DataLoaders."""

    def __init__(
        self,
        train_factory: DatasetFactory[SampleT] | None = None,
        validation_factory: DatasetFactory[SampleT] | None = None,
        test_factory: DatasetFactory[SampleT] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        multiprocessing_context: str | None = None,
        drop_last: bool = False,
        batch_format: Literal["default", "tensordict"] = "default",
        train_loader_config: DeltaDataLoaderConfig | None = None,
        validation_loader_config: DeltaDataLoaderConfig | None = None,
        test_loader_config: DeltaDataLoaderConfig | None = None,
    ) -> None:
        """Create a Lightning data module for stage-specific Delta datasets.

        Args:
            train_factory: Factory that creates the training dataset.
            validation_factory: Factory that creates the validation dataset.
            test_factory: Factory that creates the test dataset.
            batch_size: Number of samples per batch.
            num_workers: Number of worker processes used to load data.
            pin_memory: Whether to copy batches into pinned memory.
            persistent_workers: Whether to keep workers alive between iterations.
            prefetch_factor: Number of batches loaded in advance by each worker.
            multiprocessing_context: Multiprocessing context used by workers.
            drop_last: Whether to drop the final incomplete batch.
            batch_format: Batch collation format.
            train_loader_config: Optional training DataLoader overrides.
            validation_loader_config: Optional validation DataLoader overrides.
            test_loader_config: Optional test DataLoader overrides.
        """
        super().__init__()
        self.train_factory = train_factory
        self.validation_factory = validation_factory
        self.test_factory = test_factory
        self.loader_config = DeltaDataLoaderConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
            drop_last=drop_last,
            batch_format=batch_format,
        )
        self.train_loader_config = train_loader_config
        self.validation_loader_config = validation_loader_config
        self.test_loader_config = test_loader_config
        self.data_train: DeltaIterableDataset[SampleT] | None = None
        self.data_val: DeltaIterableDataset[SampleT] | None = None
        self.data_test: DeltaIterableDataset[SampleT] | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create datasets required for the requested training stage.

        Args:
            stage: Stage to prepare, or None to prepare all stages.
        """
        if stage in {None, "fit"}:
            self.data_train = self._create_dataset(self.train_factory, "train")
            self.data_val = self._create_dataset(self.validation_factory, "validation")
        if stage in {None, "test"}:
            self.data_test = self._create_dataset(self.test_factory, "test")

    def train_dataloader(self) -> DataLoader[SampleT]:
        """Return the training DataLoader."""
        return self._dataloader(self._require_dataset(self.data_train, "train"), self.train_loader_config)

    def val_dataloader(self) -> DataLoader[SampleT]:
        """Return the validation DataLoader."""
        return self._dataloader(self._require_dataset(self.data_val, "validation"), self.validation_loader_config)

    def test_dataloader(self) -> DataLoader[SampleT]:
        """Return the test DataLoader."""
        return self._dataloader(self._require_dataset(self.data_test, "test"), self.test_loader_config)

    @staticmethod
    def _create_dataset(factory: DatasetFactory[SampleT] | None, stage: str) -> DeltaIterableDataset[SampleT]:
        if factory is None:
            raise InvalidDatasetConfigurationError(f"No {stage} dataset factory was provided.")
        return factory if isinstance(factory, DeltaIterableDataset) else factory()

    @staticmethod
    def _require_dataset(dataset: DeltaIterableDataset[SampleT] | None, stage: str) -> DeltaIterableDataset[SampleT]:
        if dataset is None:
            raise InvalidDatasetConfigurationError(f"Call setup('{stage}') before creating its DataLoader.")
        return dataset

    def _dataloader(
        self, dataset: DeltaIterableDataset[SampleT], config: DeltaDataLoaderConfig | None
    ) -> DeltaDataLoader[SampleT]:
        config = config or self.loader_config
        return DeltaDataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            multiprocessing_context=config.multiprocessing_context,
            drop_last=config.drop_last,
            batch_format=config.batch_format,
        )
