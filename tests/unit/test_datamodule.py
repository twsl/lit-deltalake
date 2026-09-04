from typing import TypedDict

import pytest

from lit_deltalake.dataloaders import DeltaDataLoader
from lit_deltalake.datamodule import DeltaDataLoaderConfig, DeltaDataModule
from lit_deltalake.datasets import DeltaIterableDataset
from lit_deltalake.readers import DeltaReader
from lit_deltalake.readers.types import DeltaScan
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError


class Sample(TypedDict):
    feature: object


class EmptyReader(DeltaReader):
    def scan(self, request: DeltaScan):
        del request
        return iter(())


def make_dataset() -> DeltaIterableDataset[Sample]:
    return DeltaIterableDataset("memory://table", EmptyReader(), DeltaScan("memory://table", columns=("feature",)))


def test_datamodule_creates_stage_datasets_and_loader() -> None:
    module = DeltaDataModule[Sample](train_factory=make_dataset, validation_factory=make_dataset, batch_size=4)

    module.setup("fit")

    assert isinstance(module.train_dataloader(), DeltaDataLoader)
    assert isinstance(module.val_dataloader(), DeltaDataLoader)


def test_datamodule_accepts_dataset_instances() -> None:
    train_dataset = make_dataset()
    validation_dataset = make_dataset()
    module = DeltaDataModule[Sample](
        train_factory=train_dataset,
        validation_factory=validation_dataset,
        batch_size=4,
    )

    module.setup("fit")

    assert module.data_train is train_dataset
    assert module.data_val is validation_dataset


def test_datamodule_requires_setup_before_loader() -> None:
    module = DeltaDataModule(train_factory=make_dataset)

    with pytest.raises(InvalidDatasetConfigurationError, match="setup"):
        module.train_dataloader()


def test_datamodule_applies_stage_loader_overrides() -> None:
    module = DeltaDataModule[Sample](
        train_factory=make_dataset,
        validation_factory=make_dataset,
        batch_size=4,
        num_workers=0,
        drop_last=True,
        train_loader_config=DeltaDataLoaderConfig(batch_size=2, drop_last=True),
        validation_loader_config=DeltaDataLoaderConfig(batch_size=4, drop_last=False, batch_format="tensordict"),
    )

    module.setup("fit")

    train_loader = module.train_dataloader()
    validation_loader = module.val_dataloader()
    assert train_loader.batch_size == 2
    assert train_loader.drop_last is True
    assert validation_loader.batch_size == 4
    assert validation_loader.drop_last is False
    assert getattr(validation_loader.collate_fn, "__name__", None) == "tensordict_collate"
