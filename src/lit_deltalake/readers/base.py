from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa

from lit_deltalake.readers.types import DeltaScan


@dataclass(frozen=True, slots=True)
class ReaderCapabilities:
    """Operations supported by a Delta reader implementation.

    ``supports_source_sharding`` is false when ``scan_shard`` cannot restrict
    the source query to assigned data, as with readers that collect a full
    Spark or Sail query. In that case, the dataset applies row-level sharding
    after scanning each batch.
    """

    supports_worker_processes: bool = True
    supports_keyed_lookup: bool = False
    exact_length: bool = False
    supports_source_sharding: bool = False


class DeltaReader(ABC):
    """Read a Delta table scan as Arrow record batches."""

    capabilities = ReaderCapabilities()

    @abstractmethod
    def scan(self, request: DeltaScan) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches for a configured Delta table scan."""

    def scan_shard(self, request: DeltaScan, shard_id: int, shard_count: int) -> Iterator[pa.RecordBatch]:
        """Yield record batches assigned to one distributed shard.

        Readers without source-level sharding fall back to a full scan; callers
        must apply row-level sharding to that result.
        """
        del shard_id, shard_count
        yield from self.scan(request)

    def length(self, request: DeltaScan) -> int | None:
        """Return exact scan cardinality when it is cheap and reliable."""
        del request
        return None
