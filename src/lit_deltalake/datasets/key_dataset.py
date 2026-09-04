from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import cast

from torch.utils.data import Dataset

from lit_deltalake.datasets.base import BaseDeltaDataset
from lit_deltalake.datasets.conversion import RowType
from lit_deltalake.readers.base import DeltaReader
from lit_deltalake.readers.types import ColumnSpec, DeltaScan, Filter, Record, SampleT, Transform
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError, UnsupportedReaderOperationError


class DeltaKeyDataset(BaseDeltaDataset[SampleT], Dataset[SampleT]):
    """Map explicit stable keys to individual Delta table rows."""

    def __init__(
        self,
        reader: DeltaReader,
        scan: DeltaScan,
        id_column: str,
        keys: Sequence[str | int],
        column_specs: tuple[ColumnSpec, ...] | None = None,
        row_transform: Transform | None = None,
        row_type: RowType | None = None,
        transform: Transform | None = None,
        cache_size: int = 1_024,
    ) -> None:
        """Create a map-style dataset for stable keyed Delta table rows.

        Args:
            reader: Reader used to scan the table.
            scan: Read configuration for the versioned table snapshot.
            id_column: Source column containing each row's stable key.
            keys: Stable keys identifying rows to retrieve.
            column_specs: Optional mappings from source columns to sample fields.
            row_transform: Optional transform applied to each selected row.
            row_type: Optional dataclass type used to construct samples.
            transform: Optional transform applied to each selected row.
            cache_size: Maximum number of version-pinned records retained for repeated lookups.

        Raises:
            InvalidDatasetConfigurationError: If dataset configuration is invalid.
            UnsupportedReaderOperationError: If the reader does not support keyed lookup.
        """
        if not reader.capabilities.supports_keyed_lookup:
            raise UnsupportedReaderOperationError(f"{type(reader).__name__} does not support keyed lookup.")
        if scan.version is None:
            raise InvalidDatasetConfigurationError("DeltaKeyDataset requires an explicit Delta table version.")
        if not id_column:
            raise InvalidDatasetConfigurationError("id_column must not be empty.")
        if len(set(keys)) != len(keys):
            raise InvalidDatasetConfigurationError("DeltaKeyDataset keys must be unique.")
        if scan.columns is None:
            raise InvalidDatasetConfigurationError("DeltaKeyDataset requires explicit scan columns.")
        if cache_size < 0:
            raise InvalidDatasetConfigurationError("cache_size must not be negative.")
        self.id_column = id_column
        self.keys = tuple(keys)
        self.cache_size = cache_size
        self._record_cache: OrderedDict[str | int, Record] = OrderedDict()
        super().__init__(reader, scan, column_specs, row_transform, row_type, transform)
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate keyed dataset-specific configuration."""
        if self.column_specs is None:
            raise InvalidDatasetConfigurationError("DeltaKeyDataset requires explicit column specs.")

    def __len__(self) -> int:
        """Return the number of requested keys."""
        return len(self.keys)

    def __getitem__(self, index: int) -> SampleT:
        """Return the sample identified by a positional key index.

        Args:
            index: Position in the configured key sequence.
        """
        key = self.keys[index]
        return self.transform_record(self._lookup_records((key,))[key])

    def __getitems__(self, indices: Sequence[int]) -> list[SampleT]:
        """Return samples for multiple positional key indices in request order.

        Args:
            indices: Positions in the configured key sequence.
        """
        if not indices:
            return []
        keys = tuple(self.keys[index] for index in indices)
        rows_by_key = self._lookup_records(keys)
        return [self.transform_record(rows_by_key[key]) for key in keys]

    def _lookup_records(self, keys: Sequence[str | int]) -> dict[str | int, Record]:
        records: dict[str | int, Record] = {}
        missing_keys: list[str | int] = []
        for key in keys:
            cached_record = self._record_cache.get(key)
            if cached_record is None:
                if key not in missing_keys:
                    missing_keys.append(key)
                continue
            self._record_cache.move_to_end(key)
            records[key] = cached_record
        if not missing_keys:
            return records

        fetched_records: dict[str | int, Record] = {}
        for record in self._records(
            (Filter(self.id_column, lambda column, value: column.isin(value), tuple(missing_keys)),)
        ):
            key = cast(str | int, record[self.id_column])
            if key in fetched_records:
                raise KeyError(f"Expected exactly one row for {self.id_column}={key!r}; found multiple.")
            fetched_records[key] = record
        if set(fetched_records) != set(missing_keys):
            raise KeyError("One or more requested Delta keys were not found.")
        for key in missing_keys:
            record = fetched_records[key]
            records[key] = record
            self._cache_record(key, record)
        return records

    def _cache_record(self, key: str | int, record: Record) -> None:
        if self.cache_size == 0:
            return
        self._record_cache[key] = record
        self._record_cache.move_to_end(key)
        if len(self._record_cache) > self.cache_size:
            self._record_cache.popitem(last=False)

    def _records(self, filters: tuple[Filter, ...]) -> Iterator[Record]:
        request = DeltaScan(
            table_uri=self.scan.table_uri,
            columns=self.scan.columns,
            filters=self.scan.filters + filters,
            version=self.scan.version,
            storage_options=self.scan.storage_options,
            batch_size=self.scan.batch_size,
        )
        for batch in self.reader.scan(request):
            yield from cast(list[Record], batch.to_pylist())
