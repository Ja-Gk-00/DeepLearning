from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Union

@dataclass
class TrainingConfig:
    image_size: int = 64
    train_batch_size: int = 32
    eval_batch_size: int = 16
    num_epochs: int = 75
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 30
    mixed_precision: str = 'fp16'
    output_dir: str = 'Unet_Logger'
    data_dir: str = '../Data/Raw/primary_data/cats/Data/'
    seed: int = 2137
    fraction: float = 0.1

    # Pamameters for noise scheduler
    num_train_timesteps: int = 1000
    # beta_start should be lower than beta_end
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear" # vals {"linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"}
    variance_type: str = "fixed_small" # vals {"fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log", "learned", "learned_range"}

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "TrainingConfig":
        with open(json_path, "r") as f:
            data = json.load(f)
        valid = {f.name for f in fields(cls)}
        init_kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(**init_kwargs)
