from typing import Any

import pytest

from lit_deltalake.dataloaders import DeltaDataLoader
from lit_deltalake.datasets import DeltaIterableDataset
from lit_deltalake.readers import DeltaReader, ReaderCapabilities
from lit_deltalake.readers.types import DeltaScan
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError


class EmptyReader(DeltaReader):
    def scan(self, request: DeltaScan):
        del request
        return iter(())


class SingleProcessReader(EmptyReader):
    capabilities = ReaderCapabilities(supports_worker_processes=False)


def make_dataset(reader: DeltaReader | None = None) -> DeltaIterableDataset:
    return DeltaIterableDataset(
        "memory://table", reader or EmptyReader(), DeltaScan("memory://table", columns=("feature",))
    )


def test_dataloader_forwards_configuration() -> None:
    loader = DeltaDataLoader(
        make_dataset(),
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="spawn",
        drop_last=True,
    )

    assert loader.batch_size == 4
    assert loader.num_workers == 1
    assert loader.pin_memory is True
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
    assert loader.multiprocessing_context is not None
    assert loader.drop_last is True


def test_dataloader_allows_dataloader_overrides() -> None:
    def collate_fn(batch: list[Any]) -> list[Any]:
        return batch

    def worker_init_fn(worker_id: int) -> None:
        del worker_id

    loader = DeltaDataLoader(
        make_dataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        timeout=4,
        worker_init_fn=worker_init_fn,
        pin_memory_device="cpu",
        in_order=False,
    )

    assert loader.timeout == 4
    assert loader.worker_init_fn is worker_init_fn
    assert loader.collate_fn is collate_fn
    assert loader.pin_memory_device == "cpu"
    assert loader.in_order is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"num_workers": -1}, "num_workers"),
        ({"persistent_workers": True}, "persistent_workers"),
        ({"prefetch_factor": 2}, "prefetch_factor"),
    ],
)
def test_dataloader_rejects_invalid_configuration(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match=message):
        DeltaDataLoader(make_dataset(), **kwargs)


def test_dataloader_rejects_unsupported_worker_reader() -> None:
    with pytest.raises(InvalidDatasetConfigurationError, match="SingleProcessReader"):
        DeltaDataLoader(make_dataset(SingleProcessReader()), num_workers=1)
