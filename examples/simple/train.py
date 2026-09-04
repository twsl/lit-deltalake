from argparse import ArgumentParser
from pathlib import Path
from typing import cast

import lightning
import torch
from torch import nn

from lit_deltalake.dataloaders import create_pytorch_dataloader
from lit_deltalake.readers.types import ColumnSpec, Filter

FEATURE_COLUMNS = ("amount_usd", "deposit_amount", "current_balance", "is_test")
LABEL_COLUMN = "event_type"
LABELS = ("deposit", "purchase", "transfer", "withdrawal")
TRAIN_FILTERS = (Filter("is_test", lambda column, value: column == value, False),)
TEST_FILTERS = (Filter("is_test", lambda column, value: column == value, True),)
DEFAULT_TABLE_URI = Path(__file__).resolve().parents[2] / "data" / "sample_events"


def _to_float(value: object) -> float:
    return float(cast(float, value))


def _event_type_to_index(value: object) -> int:
    return LABELS.index(cast(str, value))


class EventClassifier(lightning.LightningModule):
    """Classify 100 sample financial transactions from numeric features."""

    def __init__(self, learning_rate: float = 1e-3) -> None:
        """Create an event classifier."""
        super().__init__()
        self.model = nn.Linear(len(FEATURE_COLUMNS), len(LABELS))
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of feature vectors."""
        return self.model(features)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        features = torch.stack(tuple(batch[column].float() for column in FEATURE_COLUMNS), dim=1)
        labels = batch[LABEL_COLUMN].long()
        logits = self(features)
        loss = self.loss_function(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run one optimization step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Evaluate one validation batch."""
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Evaluate one test batch."""
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the optimizer used for training."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def _arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Train a Lightning classifier on sample Delta events.")
    parser.add_argument("--table-uri", default=str(DEFAULT_TABLE_URI))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=4096)
    return parser


def main() -> None:
    arguments = _arguments().parse_args()
    if arguments.epochs < 1:
        raise ValueError("epochs must be positive.")
    columns = FEATURE_COLUMNS + (LABEL_COLUMN,)
    column_specs = tuple(ColumnSpec(column, transform=_to_float) for column in FEATURE_COLUMNS) + (
        ColumnSpec(LABEL_COLUMN, transform=_event_type_to_index),
    )
    loader = create_pytorch_dataloader(
        arguments.table_uri,
        columns=columns,
        column_specs=column_specs,
        filters=TRAIN_FILTERS,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        shuffle_buffer_size=arguments.shuffle_buffer_size,
    )
    test_loader = create_pytorch_dataloader(
        arguments.table_uri,
        columns=columns,
        column_specs=column_specs,
        filters=TEST_FILTERS,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
    )
    trainer = lightning.Trainer(
        max_epochs=arguments.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    model = EventClassifier(learning_rate=arguments.learning_rate)
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=test_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
