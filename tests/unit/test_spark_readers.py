import sys
from types import ModuleType
from typing import Any, cast

import pytest

from lit_deltalake.readers.sail import start_sail_spark_connect_server
import lit_deltalake.readers.spark as spark_module
from lit_deltalake.readers.spark import SparkDeltaReader
from lit_deltalake.readers.types import DeltaScan, Filter
from lit_deltalake.utils.errors import (
    InvalidDatasetConfigurationError,
    UnsupportedReaderOperationError,
)


class FakeSparkSession:
    active_session: "FakeSparkSession | None" = None

    @classmethod
    def getActiveSession(cls) -> "FakeSparkSession | None":  # noqa: N802
        return cls.active_session


def install_pyspark(monkeypatch: pytest.MonkeyPatch) -> None:
    pyspark = ModuleType("pyspark")
    sql = ModuleType("pyspark.sql")
    cast(Any, sql).SparkSession = FakeSparkSession
    cast(Any, pyspark).sql = sql
    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)


class FakeColumn:
    def __init__(self, name: str) -> None:
        self.name = name
        self.operations: list[tuple[str, object]] = []

    def _compare(self, operator: str, value: object) -> bool:
        self.operations.append((operator, value))
        return True

    def __eq__(self, value: object) -> bool:
        return self._compare("=", value)

    def __ne__(self, value: object) -> bool:
        return self._compare("!=", value)

    def __lt__(self, value: object) -> bool:
        return self._compare("<", value)

    def __le__(self, value: object) -> bool:
        return self._compare("<=", value)

    def __gt__(self, value: object) -> bool:
        return self._compare(">", value)

    def __ge__(self, value: object) -> bool:
        return self._compare(">=", value)

    def isin(self, value: object) -> bool:
        return self._compare("in", value)


def test_spark_filter_translation_calls_filter_function(monkeypatch: pytest.MonkeyPatch) -> None:
    install_pyspark(monkeypatch)
    functions = ModuleType("pyspark.sql.functions")
    cast(Any, functions).col = FakeColumn
    cast(Any, sys.modules["pyspark.sql"]).functions = functions
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)
    filter_clause = Filter("value", lambda column, value: column > value, 1)

    assert SparkDeltaReader._to_spark_filter(filter_clause) is True


def test_spark_reader_forwards_snapshot_and_storage_options(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReader:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def format(self, value: str) -> "FakeReader":
            self.calls.append(("format", value))
            return self

        def option(self, name: str, value: object) -> "FakeReader":
            self.calls.append((name, value))
            return self

        def options(self, **values: str) -> "FakeReader":
            self.calls.extend(values.items())
            return self

        def load(self, table_uri: str) -> "FakeDataFrame":
            self.calls.append(("load", table_uri))
            return FakeDataFrame()

    class FakeDataFrame:
        def toLocalIterator(self, *, prefetchPartitions: bool) -> list["FakeRow"]:  # noqa: N802, N803
            if prefetchPartitions:
                raise AssertionError("Spark reader must not prefetch partitions.")
            return [FakeRow({"value": 1}), FakeRow({"value": 2})]

    class FakeRow:
        def __init__(self, value: dict[str, object]) -> None:
            self.value = value

        def asDict(self, *, recursive: bool) -> dict[str, object]:  # noqa: N802
            assert recursive
            return self.value

    fake_reader = FakeReader()
    session = cast(Any, type("FakeSession", (), {"read": fake_reader})())
    monkeypatch.setattr(SparkDeltaReader, "_validate_pyspark", staticmethod(lambda: None))

    assert list(
        SparkDeltaReader(session).scan(
            DeltaScan("table", version=7, storage_options={"fs.s3a.endpoint": "local"}, batch_size=11)
        )
    )[0].to_pylist() == [{"value": 1}, {"value": 2}]
    assert fake_reader.calls == [
        ("format", "delta"),
        ("versionAsOf", "7"),
        ("fs.s3a.endpoint", "local"),
        ("load", "table"),
    ]


def test_spark_reader_yields_bounded_batches_without_consuming_remaining_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed_rows: list[int] = []

    class FakeRow:
        def __init__(self, value: int) -> None:
            self.value = value

        def asDict(self, *, recursive: bool) -> dict[str, int]:  # noqa: N802
            assert recursive
            return {"value": self.value}

    class FakeDataFrame:
        def toLocalIterator(self, *, prefetchPartitions: bool):  # noqa: N802, N803
            if prefetchPartitions:
                raise AssertionError("Spark reader must not prefetch partitions.")
            for value in range(5):
                consumed_rows.append(value)
                yield FakeRow(value)

    fake_reader = cast(
        Any, type("FakeReader", (), {"format": lambda self, _: self, "load": lambda self, _: FakeDataFrame()})()
    )
    session = cast(Any, type("FakeSession", (), {"read": fake_reader})())
    monkeypatch.setattr(SparkDeltaReader, "_validate_pyspark", staticmethod(lambda: None))

    batches = SparkDeltaReader(session).scan(DeltaScan("table", batch_size=2))

    assert next(batches).to_pylist() == [{"value": 0}, {"value": 1}]
    assert consumed_rows == [0, 1]
    assert next(batches).to_pylist() == [{"value": 2}, {"value": 3}]
    assert list(batches)[0].to_pylist() == [{"value": 4}]


def test_spark_reader_requires_local_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataFrame:
        pass

    fake_reader = cast(
        Any, type("FakeReader", (), {"format": lambda self, _: self, "load": lambda self, _: FakeDataFrame()})()
    )
    session = cast(Any, type("FakeSession", (), {"read": fake_reader})())
    monkeypatch.setattr(SparkDeltaReader, "_validate_pyspark", staticmethod(lambda: None))

    with pytest.raises(UnsupportedReaderOperationError, match="toLocalIterator"):
        list(SparkDeltaReader(session).scan(DeltaScan("table")))


def test_spark_reader_rejects_multi_rank_distributed_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    session = cast(Any, type("FakeSession", (), {"read": object()})())
    monkeypatch.setattr(SparkDeltaReader, "_validate_pyspark", staticmethod(lambda: None))
    monkeypatch.setattr(spark_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(spark_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(spark_module.distributed, "get_world_size", lambda: 2)

    with pytest.raises(InvalidDatasetConfigurationError, match="cannot source-shard"):
        list(SparkDeltaReader(session).scan(DeltaScan("table")))


def test_spark_reader_uses_active_session_when_session_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    install_pyspark(monkeypatch)
    session = FakeSparkSession()
    FakeSparkSession.active_session = session

    reader = SparkDeltaReader()

    assert reader.session is session


def test_spark_reader_requires_active_session_when_session_is_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    install_pyspark(monkeypatch)
    FakeSparkSession.active_session = None

    with pytest.raises(RuntimeError, match="No active Spark session"):
        SparkDeltaReader()


def test_start_sail_spark_connect_server_starts_foreground_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    class FakeSparkConnectServer:
        def __init__(self, port: int) -> None:
            calls.append(("init", port))

        def start(self, *, background: bool) -> None:
            calls.append(("start", background))

    pysail = ModuleType("pysail")
    spark = ModuleType("pysail.spark")
    cast(Any, spark).SparkConnectServer = FakeSparkConnectServer
    cast(Any, pysail).spark = spark
    monkeypatch.setitem(sys.modules, "pysail", pysail)
    monkeypatch.setitem(sys.modules, "pysail.spark", spark)

    server = start_sail_spark_connect_server(port=12345)

    assert isinstance(server, FakeSparkConnectServer)
    assert calls == [("init", 12345), ("start", False)]
