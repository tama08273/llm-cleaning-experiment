from dataclasses import dataclass

@dataclass
class TrainConfig:
    train_path: str = "dataset/small_dataset.txt"
    cleaner: str = "none"
    use_sp: bool = False

    epochs: int = 3
    batch_size: int = 16
    max_length: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_cuda: bool = True