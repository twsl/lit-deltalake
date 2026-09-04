from importlib import import_module
from typing import Any, Literal, cast

from lit_deltalake.dataloaders.dataloader import DeltaDataLoader
from lit_deltalake.datasets.conversion import RowType, dataclass_columns
from lit_deltalake.datasets.iterable_dataset import DeltaIterableDataset
from lit_deltalake.readers.base import DeltaReader
from lit_deltalake.readers.delta_rs import DeltaRsReader
from lit_deltalake.readers.types import (
    ColumnSpec,
    DeltaScan,
    Filter,
    SampleT,
    StorageOptions,
    Transform,
)
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError


def create_pytorch_dataloader(
    table_uri: str,
    columns: tuple[str, ...] | None = None,
    *,
    column_specs: tuple[ColumnSpec, ...] | None = None,
    filters: tuple[Filter, ...] = (),
    version: int | None = None,
    storage_options: StorageOptions | None = None,
    scan_batch_size: int = 65_536,
    row_transform: Transform | None = None,
    row_type: RowType | None = None,
    transform: Transform | None = None,
    shuffle_buffer_size: int = 0,
    shuffle_seed: int | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = None,
    drop_last: bool = False,
    batch_format: Literal["default", "tensordict"] = "default",
    reader: DeltaReader | None = None,
    backend: Literal["delta-rs", "spark", "sail"] = "delta-rs",
    spark_session: Any | None = None,
) -> DeltaDataLoader[SampleT]:
    """Create a PyTorch DataLoader backed by a Delta table scan.

    Rows are sharded across initialized PyTorch distributed ranks and DataLoader
    workers by :class:`DeltaIterableDataset`. Use ``shuffle_buffer_size`` for
    bounded streaming shuffle; PyTorch sampler shuffle is unsupported for
    iterable datasets. Spark and Sail readers collect data to client memory;
    use the default Delta-rs backend for multi-rank DDP input.
    """
    if columns is None and row_type is None:
        raise InvalidDatasetConfigurationError("Provide columns or a dataclass row_type.")
    if columns is not None and row_type is not None:
        raise InvalidDatasetConfigurationError("Provide columns or row_type, not both.")
    if reader is not None and (backend != "delta-rs" or spark_session is not None):
        raise InvalidDatasetConfigurationError("Provide reader or backend/spark_session, not both.")
    if row_type is not None:
        selected_columns = dataclass_columns(row_type)
    else:
        if columns is None:
            raise InvalidDatasetConfigurationError("Provide columns or a dataclass row_type.")
        selected_columns = columns
    scan = DeltaScan(
        table_uri=table_uri,
        columns=selected_columns,
        filters=filters,
        version=version,
        storage_options=storage_options,
        batch_size=scan_batch_size,
    )
    dataset = DeltaIterableDataset(
        table_uri,
        reader=reader or _create_reader(backend, spark_session),
        scan=scan,
        column_specs=column_specs,
        row_transform=row_transform,
        transform=transform,
        row_type=row_type,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=shuffle_seed,
    )
    return cast(
        DeltaDataLoader[SampleT],
        DeltaDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
            drop_last=drop_last,
            batch_format=batch_format,
        ),
    )


def _create_reader(backend: Literal["delta-rs", "spark", "sail"], spark_session: Any | None) -> DeltaReader:
    if backend == "delta-rs":
        if spark_session is not None:
            raise InvalidDatasetConfigurationError("spark_session requires backend='spark' or backend='sail'.")
        return DeltaRsReader()
    module_name = "spark" if backend == "spark" else "sail"
    class_name = "SparkDeltaReader" if backend == "spark" else "SailDeltaReader"
    reader_class = getattr(import_module(f"lit_deltalake.readers.{module_name}"), class_name)
    return cast(DeltaReader, reader_class(spark_session))
