from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

import ignite.distributed as idist


@dataclass
class TrainingConfig:
    # logging and data
    experiment_dir: Path = Path(".")
    output_dir: Optional[Path] = None
    progress: bool = True
    random_seed: int = 42
    dataloader_workers: int = 0
    pin_memory: bool = False
    progress: bool = True
    diskcache: Optional[Path] = None

    # optimization
    optimizer: Literal["sgd", "adamw"] = "adamw"
    scheduler: Optional[Literal["onecycle"]] = "onecycle"
    per_device_batch_size: int = 256
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    epochs: int = 30
    gradient_accumulation_steps: int = 1
    warmup_steps: float | int = 0.3
    swag: bool = False
    checkpoint: bool = True
    resume_checkpoint: Optional[Path] = None

    # evaluation
    train_eval_fraction: float = 0.1

    def __post_init__(self):
        self.batch_size = self.per_device_batch_size * idist.get_world_size()
        if self.output_dir is None:
            self.output_dir = self.experiment_dir
