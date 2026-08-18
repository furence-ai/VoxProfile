"""Configuration for Audio Gender & Age Classification."""
from dataclasses import dataclass, field
from typing import List, Optional


MODEL_CONFIGS = {
    "light": {
        "hidden_dims": [],
        "dropout": 0.1,
        "unfreeze_encoder_layers": 0,
    },
    "medium": {
        "hidden_dims": [128, 64],
        "dropout": 0.2,
        "unfreeze_encoder_layers": 4,
    },
    "heavy": {
        "hidden_dims": [512, 256, 128],
        "dropout": 0.3,
        "unfreeze_encoder_layers": 6,
    },
}


@dataclass
class TrainConfig:
    # Model
    model_size: str = "medium"                        # light / medium / heavy
    # encoder_name: NGC/HF 사전학습 모델명 OR 로컬 .nemo 파일 경로
    encoder_name: str = "titanet_large"
    encoder_out_dim: int = 192

    # Data
    train_manifest: str = "data/manifests/train.jsonl"
    val_manifest: str = "data/manifests/val.jsonl"
    test_manifest: str = "data/manifests/test.jsonl"
    sample_rate: int = 16000
    max_duration: float = 8.0
    batch_size: int = 32
    num_workers: int = 4

    # Optim
    lr_head: float = 1e-3
    lr_encoder: float = 1e-4                         # LoRA에서는 좀 더 높게 가능
    head_weight_decay: float = 1e-2
    enc_weight_decay: float = 1e-2                   # LoRA params 적어서 0.1은 과도
    epochs: int = 30
    scheduler_t_max: int = 30
    scheduler_eta_min: float = 1e-6

    # LoRA (encoder adaptation)
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    # Phase
    phase1_epochs: int = 10                          # encoder frozen
    lambda_age_phase1: float = 1.0
    lambda_age_phase2: float = 1.0

    # Regularization (Option C)
    emb_dropout: float = 0.3                         # encoder 출력 임베딩 dropout
    label_smoothing: float = 0.1                     # age CrossEntropy

    # Loss
    gender_pos_weight: float = 1.0                   # 1.0 = balanced
    huber_delta: float = 1.0

    # Early stopping
    es_patience: int = 7
    es_delta: float = 1e-4
    ckpt_path: str = "runs/best_model.pt"

    # model / 로깅
    ckpt_dir: str = "runs"
    keep_top_k: int = 3                # val_loss 기준 best K개 보관 (0이면 무제한)
    save_every_epoch: bool = True      # 매 epoch ckpt 저장 여부
    history_csv: str = "runs/history.csv"
    plot_path: str = "runs/history.png"

    # Misc
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 50

    def head_config(self) -> dict:
        return MODEL_CONFIGS[self.model_size]
