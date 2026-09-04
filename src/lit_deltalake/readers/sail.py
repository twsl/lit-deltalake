from typing import TYPE_CHECKING, override

from lit_deltalake.readers.spark import SparkDeltaReader
from lit_deltalake.utils.errors import BackendUnavailableError

if TYPE_CHECKING:
    from pysail.spark import SparkConnectServer
    from pyspark.sql import SparkSession


def start_sail_spark_connect_server(port: int = 50051) -> "SparkConnectServer":
    """Start a PySail-backed Spark Connect server in the foreground."""
    SailDeltaReader._validate_pysail()
    from pysail.spark import SparkConnectServer

    server = SparkConnectServer(port=port)
    server.start(background=False)
    return server


class SailDeltaReader(SparkDeltaReader):
    """Read Delta tables through a caller-owned Sail Spark Connect session."""

    @override
    def __init__(self, session: "SparkSession | None" = None) -> None:
        """Create a Sail-backed Delta reader.

        Args:
            session: Optional caller-owned Spark or Spark Connect session.
        """
        self._validate_pysail()
        super().__init__(session)

    @staticmethod
    def _validate_pysail() -> None:
        try:
            import pysail  # noqa: F401
        except ImportError as error:
            raise BackendUnavailableError(
                "Sail support requires `lit-deltalake[sail]`. Install the optional extra first."
            ) from error
