from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, override

import pyarrow as pa
import torch.distributed as distributed

from lit_deltalake.readers.base import DeltaReader, ReaderCapabilities
from lit_deltalake.readers.types import DeltaScan, Filter
from lit_deltalake.utils.errors import (
    BackendUnavailableError,
    InvalidDatasetConfigurationError,
    UnsupportedReaderOperationError,
)

if TYPE_CHECKING:
    from pyspark.sql import Column, SparkSession


class SparkDeltaReader(DeltaReader):
    """Read Delta tables from a caller-owned Spark or Spark Connect session."""

    capabilities = ReaderCapabilities(supports_worker_processes=False)

    def __init__(self, session: "SparkSession | None" = None) -> None:
        """Create a Spark-backed Delta reader.

        Args:
            session: Optional caller-owned Spark or Spark Connect session.
        """
        self.session = session or self._get_active_session()

    @override
    def scan(self, request: "DeltaScan") -> Iterator["pa.RecordBatch"]:
        """Yield Arrow record batches for a Spark table scan.

        Args:
            request: Configuration for the table snapshot and selected data.
        """
        self._validate_pyspark()
        if distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() > 1:
            raise InvalidDatasetConfigurationError(
                "Spark and Sail readers cannot source-shard scans for distributed DDP input; each rank would "
                "otherwise execute the full query. Use DeltaRsReader for multi-rank data loading."
            )
        reader = self.session.read.format("delta")
        if request.version is not None:
            reader = reader.option("versionAsOf", str(request.version))
        if request.storage_options:
            reader = reader.options(**dict(request.storage_options))
        dataframe = reader.load(request.table_uri)
        if request.columns is not None:
            dataframe = dataframe.select(*request.columns)
        for filter_clause in request.filters:
            dataframe = dataframe.filter(self._to_spark_filter(filter_clause))
        try:
            rows = dataframe.toLocalIterator(prefetchPartitions=False)
        except AttributeError as error:
            raise UnsupportedReaderOperationError(
                "Spark reader requires DataFrame.toLocalIterator(); use a Spark 3.4+ or Spark Connect-compatible client."
            ) from error
        records: list[dict[str, object]] = []
        for row in rows:
            records.append(row.asDict(recursive=True))
            if len(records) == request.batch_size:
                yield pa.RecordBatch.from_pylist(records)
                records = []
        if records:
            yield pa.RecordBatch.from_pylist(records)

    @staticmethod
    def _validate_pyspark() -> None:
        try:
            import pyspark.sql  # noqa: F401
        except ImportError as error:
            raise BackendUnavailableError(
                "Spark support requires `lit-deltalake[spark]`. Install the optional extra first."
            ) from error

    @classmethod
    def _get_active_session(cls) -> "SparkSession":
        cls._validate_pyspark()
        from pyspark.sql import SparkSession

        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("No active Spark session. Pass a session or create one before constructing the reader.")
        return session

    @staticmethod
    def _to_spark_filter(filter_clause: "Filter") -> Any:
        from pyspark.sql import functions as f

        column: Any = f.col(filter_clause.column)
        return filter_clause.function(column, filter_clause.value)
