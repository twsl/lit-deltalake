from abc import ABC, abstractmethod
from typing import Any, cast

from lit_deltalake.datasets.conversion import RowType, dataclass_columns, to_sample
from lit_deltalake.readers.base import DeltaReader
from lit_deltalake.readers.types import ColumnSpec, DeltaScan, Record, SampleT, Transform
from lit_deltalake.utils.errors import InvalidDatasetConfigurationError


class BaseDeltaDataset[SampleT](ABC):
    """Shared configuration and record conversion for Delta datasets."""

    reader: DeltaReader
    scan: DeltaScan
    column_specs: tuple[ColumnSpec, ...] | None
    row_type: RowType | None
    transform: Transform | None
    row_transform: Transform | None

    def __init__(
        self,
        reader: DeltaReader,
        scan: DeltaScan,
        column_specs: tuple[ColumnSpec, ...] | None,
        row_transform: Transform | None,
        row_type: RowType | None,
        transform: Transform | None,
    ) -> None:
        """Initialize shared Delta dataset configuration.

        Args:
            reader: Reader used to scan the table.
            scan: Configuration for the table snapshot.
            column_specs: Optional mappings from source columns to sample fields.
            row_transform: Optional transform applied to each selected row.
            row_type: Optional dataclass type used to construct samples.
            transform: Optional transform applied to each selected row.

        Raises:
            InvalidDatasetConfigurationError: If shared configuration is invalid.
        """
        super().__init__()
        self.reader = reader
        self.scan = scan
        self.column_specs = column_specs or self._default_column_specs(scan)
        self.transform = self._resolve_transform(row_transform, transform)
        self.row_transform = self.transform
        self.row_type = row_type
        self._validate_common_configuration()

    @staticmethod
    def _default_column_specs(scan: DeltaScan) -> tuple[ColumnSpec, ...] | None:
        """Create default column mappings from selected scan columns.

        Args:
            scan: Configuration containing selected columns.

        Returns:
            One mapping per selected column, or None when columns are not explicit.
        """
        if scan.columns is None:
            return None
        return tuple(ColumnSpec(column) for column in scan.columns)

    @staticmethod
    def _resolve_transform(row_transform: Transform | None, transform: Transform | None) -> Transform | None:
        """Resolve the backwards-compatible transform aliases.

        Args:
            row_transform: Legacy transform argument.
            transform: Preferred transform argument.

        Returns:
            The configured transform, or None.

        Raises:
            InvalidDatasetConfigurationError: If both transform arguments are provided.
        """
        if row_transform is not None and transform is not None:
            raise InvalidDatasetConfigurationError("Provide transform or row_transform, not both.")
        return transform if transform is not None else row_transform

    def _validate_common_configuration(self) -> None:
        """Validate column mappings and dataclass field alignment."""
        if self.column_specs is None:
            if self.row_type is not None:
                raise InvalidDatasetConfigurationError("row_type requires explicit column_specs or scan.columns.")
            return
        if self.scan.columns is not None and {spec.source for spec in self.column_specs} - set(self.scan.columns):
            raise InvalidDatasetConfigurationError("Every ColumnSpec source must be selected by the scan.")
        output_names = [spec.output_name for spec in self.column_specs]
        if len(set(output_names)) != len(output_names):
            raise InvalidDatasetConfigurationError("ColumnSpec output names must be unique.")
        if self.row_type is not None and tuple(output_names) != dataclass_columns(self.row_type):
            raise InvalidDatasetConfigurationError(
                "ColumnSpec output names must match row_type dataclass fields in order."
            )

    @abstractmethod
    def _validate_configuration(self) -> None:
        """Validate configuration specific to the concrete dataset protocol."""

    def transform_record(self, record: Record) -> SampleT:
        """Convert one Arrow-derived Python record into the configured sample.

        ``RecordBatch.to_pylist()`` has already converted the Arrow row into a
        Python mapping when this hook runs. Column decoders, column transforms,
        dataclass construction, and the whole-sample transform are applied
        here. Subclasses can override this method to customize per-record
        conversion.
        """
        if self.column_specs is None:
            return cast(SampleT, to_sample(record, self.row_type, self.transform))
        sample: Record = {}
        for spec in self.column_specs:
            value = record[spec.source]
            if spec.decoder is not None:
                value = spec.decoder(value)
            if spec.transform is not None:
                value = spec.transform(value)
            sample[spec.output_name] = value
        return cast(SampleT, to_sample(sample, self.row_type, self.transform))
