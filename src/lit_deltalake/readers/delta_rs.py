from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from lit_deltalake.readers.base import DeltaReader, ReaderCapabilities
from lit_deltalake.readers.types import DeltaScan, Filter
from lit_deltalake.utils.errors import BackendUnavailableError


class DeltaRsReader(DeltaReader):
    """Read Delta table snapshots through delta-rs and PyArrow."""

    capabilities = ReaderCapabilities(supports_keyed_lookup=True, exact_length=True, supports_source_sharding=True)

    def scan(self, request: DeltaScan) -> Iterator["pa.RecordBatch"]:
        """Yield Arrow record batches for a Delta-rs table scan.

        Args:
            request: Configuration for the table snapshot and selected data.
        """
        delta_table_class, ds = self._load_dependencies()
        table = delta_table_class(
            request.table_uri,
            version=request.version,
            storage_options=dict(request.storage_options or {}),
        )
        scanner = table.to_pyarrow_dataset().scanner(
            columns=list(request.columns) if request.columns is not None else None,
            filter=self._to_arrow_filter(ds, request.filters),
            batch_size=request.batch_size,
        )
        yield from scanner.to_reader()

    def scan_shard(self, request: DeltaScan, shard_id: int, shard_count: int) -> Iterator[pa.RecordBatch]:
        """Yield batches from fragments deterministically assigned to one shard."""
        if not 0 <= shard_id < shard_count:
            raise ValueError("shard_id must be non-negative and less than shard_count.")
        delta_table_class, ds = self._load_dependencies()
        table = delta_table_class(
            request.table_uri,
            version=request.version,
            storage_options=dict(request.storage_options or {}),
        )
        dataset = table.to_pyarrow_dataset()
        expression = self._to_arrow_filter(ds, request.filters)
        fragments = dataset.get_fragments(filter=expression)
        for fragment_index, fragment in enumerate(fragments):
            if fragment_index % shard_count != shard_id:
                continue
            scanner = fragment.scanner(
                columns=list(request.columns) if request.columns is not None else None,
                filter=expression,
                batch_size=request.batch_size,
            )
            yield from scanner.to_reader()

    def length(self, request: DeltaScan) -> int:
        """Return the exact number of rows selected by a Delta-rs scan."""
        delta_table_class, ds = self._load_dependencies()
        table = delta_table_class(
            request.table_uri,
            version=request.version,
            storage_options=dict(request.storage_options or {}),
        )
        return table.to_pyarrow_dataset().count_rows(filter=self._to_arrow_filter(ds, request.filters))

    @staticmethod
    def _load_dependencies() -> tuple[Any, Any]:
        try:
            from deltalake import DeltaTable
            import pyarrow.dataset as ds
        except ImportError as error:
            raise BackendUnavailableError(
                "Delta-rs support requires `lit-deltalake[delta]`. Install the optional extra first."
            ) from error
        return DeltaTable, ds

    @staticmethod
    def _to_arrow_filter(ds: Any, filters: tuple[Filter, ...]) -> Any | None:
        expression = None
        for filter_clause in filters:
            field = ds.field(filter_clause.column)
            clause = filter_clause.function(field, filter_clause.value)
            expression = clause if expression is None else expression & clause
        return expression
