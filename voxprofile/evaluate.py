"""Evaluation / inference."""
import argparse
from typing import Dict

import torch
import torch.nn as nn

from .config import TrainConfig
from .dataset import build_loader, AGE_LABELS
from .model import AgeGenderModel


@torch.no_grad()
def evaluate(cfg: TrainConfig, ckpt_path: str, manifest: str) -> Dict[str, float]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    head_cfg = cfg.head_config()
    model = AgeGenderModel(
        encoder_name=cfg.encoder_name,
        encoder_out_dim=cfg.encoder_out_dim,
        hidden_dims=head_cfg["hidden_dims"],
        dropout=head_cfg["dropout"],
        unfreeze_encoder_layers=head_cfg.get("unfreeze_encoder_layers", 0),
        emb_dropout=cfg.emb_dropout,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    loader = build_loader(manifest, cfg.batch_size, cfg.num_workers,
                          cfg.sample_rate, cfg.max_duration, shuffle=False)

    n = 0
    g_correct = 0
    age_abs = 0.0
    age_decade_correct = 0

    for audio, length, gender, age in loader:
        audio, length = audio.to(device), length.to(device)
        gender, age = gender.to(device), age.to(device)         # age: long 0..5
        g_logit, a_logits = model(audio, length)

        g_hat = (g_logit.sigmoid() > 0.5).float()
        a_hat = a_logits.argmax(dim=-1)

        bs = gender.size(0)
        n += bs
        g_correct += (g_hat == gender).sum().item()
        age_abs += (a_hat - age).abs().float().sum().item()
        age_decade_correct += (a_hat == age).sum().item()

    metrics = {
        "gender_acc": g_correct / n,
        "age_acc": age_decade_correct / n,
        "age_mae": age_abs / n,
    }
    print(metrics)
    return metrics


@torch.no_grad()
def predict_file(cfg: TrainConfig, ckpt_path: str, audio_path: str) -> Dict:
    import numpy as np
    import soundfile as sf
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    head_cfg = cfg.head_config()
    model = AgeGenderModel(
        encoder_name=cfg.encoder_name,
        encoder_out_dim=cfg.encoder_out_dim,
        hidden_dims=head_cfg["hidden_dims"],
        dropout=head_cfg["dropout"],
        unfreeze_encoder_layers=head_cfg.get("unfreeze_encoder_layers", 0),
        emb_dropout=cfg.emb_dropout,
        use_lora=cfg.use_lora,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"peak alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"peak reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    model.eval()

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != cfg.sample_rate:
        ratio = cfg.sample_rate / sr
        n_new = int(round(len(wav) * ratio))
        wav = np.interp(
            np.linspace(0, len(wav) - 1, n_new, dtype=np.float32),
            np.arange(len(wav), dtype=np.float32), wav
        ).astype(np.float32)
    audio = torch.from_numpy(wav).unsqueeze(0).to(device)
    length = torch.tensor([audio.shape[-1]], device=device)
    g_logit, a_logits = model(audio, length)
    g_id = int((g_logit.sigmoid() > 0.5).item())
    a_id = int(a_logits.argmax(dim=-1).item())     # 0..5
    a_probs = a_logits.softmax(dim=-1).squeeze(0).tolist()
    return {
        "gender": "female" if g_id == 1 else "male",
        "age": AGE_LABELS[a_id],
        "gender_prob": round(g_logit.sigmoid().item(), 4),
        "age_probs": {AGE_LABELS[i]: round(p, 4) for i, p in enumerate(a_probs)},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--audio", default=None)
    args = p.parse_args()

    cfg = TrainConfig()
    if args.audio:
        print(predict_file(cfg, args.ckpt, args.audio))
    else:
        evaluate(cfg, args.ckpt, args.manifest or cfg.test_manifest)
