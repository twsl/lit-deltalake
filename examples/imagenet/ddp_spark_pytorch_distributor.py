"""Train ImageNet-style Delta tables with local PyTorch Lightning.

Tables require binary ``content`` images and integer ``object_id`` labels. Use
``--class-map`` when object IDs are not zero-based class indexes.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import lightning as lightning
import torch
from torch import nn
from torchvision.io import decode_image

from lit_deltalake.datamodule import DeltaDataLoaderConfig, DeltaDataModule
from lit_deltalake.datasets import DeltaIterableDataset
from lit_deltalake.readers import DeltaRsReader
from lit_deltalake.readers.types import ColumnSpec, DeltaScan


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_uri: str
    validation_uri: str
    batch_size: int
    epochs: int
    learning_rate: float
    num_classes: int
    num_workers: int
    version: int | None
    class_map: dict[int, int] | None


def _decode_image(value: bytes) -> Any:
    return decode_image(torch.frombuffer(bytearray(value), dtype=torch.uint8))


def _image_transform() -> Any:
    from torchvision import transforms

    return transforms.Compose(
        (
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
    )


def _transform_label(value: int, class_map: dict[int, int] | None) -> int:
    label = int(value)
    if class_map is None:
        return label
    try:
        return class_map[label]
    except KeyError as error:
        raise ValueError(f"object_id {label!r} is absent from the class map.") from error


def _dataset_factory(table_uri: str, config: TrainingConfig, shuffle: bool) -> DeltaIterableDataset[Any]:
    return DeltaIterableDataset(
        table_uri,
        DeltaRsReader(),
        DeltaScan(table_uri, columns=("content", "object_id"), version=config.version),
        column_specs=(
            ColumnSpec(
                "content",
                decoder=lambda value: _decode_image(cast(bytes, value)),
                transform=_image_transform(),
            ),
            ColumnSpec(
                "object_id",
                name="label",
                transform=lambda value: _transform_label(cast(int, value), config.class_map),
            ),
        ),
        shuffle_buffer_size=config.batch_size * 16 if shuffle else 0,
    )


class ImageNetDataModule(DeltaDataModule[Any]):
    """Create ImageNet Delta datasets for local training."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(
            train_factory=lambda: _dataset_factory(config.train_uri, config, shuffle=True),
            validation_factory=lambda: _dataset_factory(config.validation_uri, config, shuffle=False),
            test_factory=lambda: _dataset_factory(config.validation_uri, config, shuffle=False),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            multiprocessing_context="spawn" if config.num_workers else None,
            train_loader_config=DeltaDataLoaderConfig(drop_last=True),
            validation_loader_config=DeltaDataLoaderConfig(drop_last=False),
            test_loader_config=DeltaDataLoaderConfig(drop_last=False),
        )
        self.num_classes = config.num_classes


class ImageNetClassificationModel(lightning.LightningModule):
    """Fine-tune ImageNet-pretrained ResNet50 for local training."""

    def __init__(self, num_classes: int, learning_rate: float) -> None:
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        self.save_hyperparameters()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.learning_rate = learning_rate

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        del batch_index
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        del batch_index
        self._step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        del batch_index
        self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch["content"])
        labels = batch["label"].to(dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=stage != "train")
        self.log(f"{stage}_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def _load_class_map(path: Path | None) -> dict[int, int] | None:
    if path is None:
        return None
    values = json.loads(path.read_text())
    try:
        return {int(object_id): int(class_index) for object_id, class_index in values.items()}
    except AttributeError as error:
        raise TypeError("class map must contain a JSON object of object_id-to-class mappings.") from error


def _arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Train ImageNet-style Delta tables with local PyTorch Lightning.")
    parser.add_argument("train_uri")
    parser.add_argument("validation_uri")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--version", type=int)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    return parser


def main() -> None:
    arguments: Namespace = _arguments().parse_args()
    if arguments.epochs < 1 or arguments.num_classes < 2 or arguments.num_workers < 0:
        raise ValueError("epochs must be positive, num_classes at least 2, and num_workers non-negative.")
    config = TrainingConfig(
        train_uri=arguments.train_uri,
        validation_uri=arguments.validation_uri,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        num_classes=arguments.num_classes,
        num_workers=arguments.num_workers,
        version=arguments.version,
        class_map=_load_class_map(arguments.class_map),
    )
    torch.set_float32_matmul_precision("medium")
    data_module = ImageNetDataModule(config)
    trainer = lightning.Trainer(
        accelerator=arguments.accelerator,
        devices=arguments.devices,
        strategy=arguments.strategy,
        max_epochs=config.epochs,
    )
    model = ImageNetClassificationModel(data_module.num_classes, config.learning_rate)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
