# Data Flow

Delta readers return Arrow `RecordBatch` objects. The dataset converts their
rows into samples, and PyTorch later collates samples into the final batches
consumed by training code.

## Terms

| Term           | Meaning                                                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Read batch** | An Arrow `RecordBatch` returned by a reader. Its size is controlled by `DeltaScan.batch_size`. It is an internal read unit, not the final tensor batch. |
| **Batch**      | The final object yielded by the PyTorch `DataLoader`. It can contain stacked tensors, a dataclass whose fields are tensors, or a `TensorDict`.          |
| **Shard**      | Data assigned to one DDP rank and DataLoader worker. Source-sharded readers restrict the scan; other readers scan broadly and filter rows afterward.    |
| **Row**        | One record inside a read batch. `RecordBatch.to_pylist()` converts it into a Python mapping, which becomes one dataset sample.                          |
| **Element**    | One field or value within a row, such as `feature` or `label`. `ColumnSpec.decoder` and `ColumnSpec.transform` operate on elements.                     |
| **Sample**     | One row after dataset conversion. It can be a mapping or dataclass, and may contain tensors if a transform creates them.                                |

## Execution Order

```text
Delta table
  -> reader scan
  -> read batch (Arrow RecordBatch)
  -> shard selection
  -> row
  -> Python record
  -> element decoding/transforms
  -> sample transform
  -> samples
  -> PyTorch DataLoader batch
```

`DeltaIterableDataset.transform_shard()` handles read-batch and shard
processing. `BaseDeltaDataset.transform_record()` handles row-to-sample
conversion. PyTorch's `default_collate` converts compatible sample fields into
tensors when the DataLoader creates its output batch.

`scan_batch_size` and `batch_size` are different:

- `scan_batch_size` controls the number of rows in each reader-side read batch.
- `batch_size` controls the number of samples in each final DataLoader batch.
