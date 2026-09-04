from collections.abc import Iterator
import random
from typing import cast

import pyarrow as pa
import torch.distributed as distributed
from torch.utils.data import IterableDataset, get_worker_info

from lit_deltalake.datasets.base import BaseDeltaDataset
from lit_deltalake.datasets.conversion import RowType
from lit_deltalake.readers.base import DeltaReader
from lit_deltalake.readers.types import ColumnSpec, DeltaScan, Record, SampleT, Transform
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError


class DeltaIterableDataset(BaseDeltaDataset[SampleT], IterableDataset[SampleT]):
    """Stream transformed rows from a Delta table through a reader implementation."""

    def __init__(
        self,
        table_uri: str,
        reader: DeltaReader,
        scan: DeltaScan | None = None,
        column_specs: tuple[ColumnSpec, ...] | None = None,
        row_transform: Transform | None = None,
        row_type: RowType | None = None,
        transform: Transform | None = None,
        shuffle_buffer_size: int = 0,
        shuffle_seed: int | None = None,
    ) -> None:
        """Create a streaming dataset for a Delta table.

        Args:
            table_uri: URI of the Delta table.
            reader: Reader used to scan the table.
            scan: Optional configuration for the table snapshot.
            column_specs: Optional mappings from source columns to sample fields.
            row_transform: Optional transform applied to each selected row.
            row_type: Optional dataclass type used to construct samples.
            transform: Optional transform applied to each selected row.
            shuffle_buffer_size: Maximum number of rows held for streaming shuffle.
            shuffle_seed: Optional seed for streaming shuffle.

        Raises:
            InvalidDatasetConfigurationError: If dataset configuration is invalid.
        """
        self.scan = scan or DeltaScan(table_uri)
        if scan is not None and scan.table_uri != table_uri:
            raise InvalidDatasetConfigurationError("scan.table_uri must match table_uri.")
        super().__init__(reader, self.scan, column_specs, row_transform, row_type, transform)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self._epoch = 0
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate streaming-specific configuration."""
        if self.shuffle_buffer_size < 0:
            raise InvalidDatasetConfigurationError("shuffle_buffer_size must not be negative.")

    def __iter__(self) -> Iterator[SampleT]:
        """Yield transformed rows assigned to the current rank and worker."""
        worker_info = get_worker_info()
        if worker_info is not None and not self.reader.capabilities.supports_worker_processes:
            raise InvalidDatasetConfigurationError(
                f"{type(self.reader).__name__} cannot run in DataLoader worker processes; use num_workers=0."
            )

        worker_id = worker_info.id if worker_info else 0
        worker_count = worker_info.num_workers if worker_info else 1
        rank = distributed.get_rank() if distributed.is_available() and distributed.is_initialized() else 0
        world_size = distributed.get_world_size() if distributed.is_available() and distributed.is_initialized() else 1
        shard_id = rank * worker_count + worker_id
        shard_count = world_size * worker_count
        rows = self._iter_rows(shard_id, shard_count)
        if self.shuffle_buffer_size:
            yield from self._shuffle(rows, shard_id)
            return
        yield from rows

    def __len__(self) -> int:
        """Return the exact number of rows in the configured scan.

        Raises:
            TypeError: If the reader cannot provide an exact length.
        """
        exact_length = self.reader.length(self.scan)
        if exact_length is None:
            raise TypeError(f"{type(self.reader).__name__} does not provide an exact dataset length.")
        return exact_length

    def set_epoch(self, epoch: int) -> None:
        """Set epoch used to derive the seeded streaming shuffle order.

        Args:
            epoch: Non-negative epoch number.

        Raises:
            ValueError: If epoch is negative.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self._epoch = epoch

    def _iter_rows(self, shard_id: int, shard_count: int) -> Iterator[SampleT]:
        row_index = 0
        for batch in self.reader.scan_shard(self.scan, shard_id, shard_count):
            yield from self.transform_shard(batch, shard_id, shard_count, row_index)
            row_index += batch.num_rows

    def transform_shard(
        self,
        read_batch: pa.RecordBatch,
        shard_id: int,
        shard_count: int,
        row_index: int,
    ) -> Iterator[SampleT]:
        """Convert one scanned Arrow read batch into samples for this shard.

        Readers with ``supports_source_sharding=True`` have already limited
        ``read_batch`` to this shard. Otherwise, this method applies row-level
        round-robin sharding using ``row_index``. Override it to customize
        read-batch-level conversion while retaining the reader's sharding
        contract.
        """
        if self.reader.capabilities.supports_source_sharding:
            records = read_batch.to_pylist()
        else:
            first_row = (shard_id - row_index) % shard_count
            indices = list(range(first_row, read_batch.num_rows, shard_count))
            try:
                records = read_batch.take(indices).to_pylist()
            except Exception as error:
                if type(error).__name__ != "ArrowNotImplementedError":
                    raise
                records = [read_batch.slice(index, 1).to_pylist()[0] for index in indices]
        for record in records:
            yield self.transform_record(cast(Record, record))

    def _shuffle(self, rows: Iterator[SampleT], worker_id: int) -> Iterator[SampleT]:
        seed = f"{self.shuffle_seed}:{self._epoch}:{worker_id}" if self.shuffle_seed is not None else None
        random_source = random.Random(seed)  # noqa: S311  # nosec B311
        buffer: list[SampleT] = []
        for row in rows:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(row)
                continue
            index = random_source.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        random_source.shuffle(buffer)
        yield from buffer
