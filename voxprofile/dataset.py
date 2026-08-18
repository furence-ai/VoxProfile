"""Audio dataset for gender & age classification.

Manifest format (JSONL), one sample per line:
    {"audio": "/path/to/file.wav", "gender": "male"|"female", "age": "20대"}
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset


GENDER2ID = {"male": 0, "female": 1, "m": 0, "f": 1, "남": 0, "여": 1}

AGE_LABELS = ["10대", "20대", "30대", "40대", "50대", "60대+"]
AGE2ID = {label: i + 1 for i, label in enumerate(AGE_LABELS)}   # 1..6


def parse_gender(v) -> int:
    if isinstance(v, int):
        return v
    return GENDER2ID[str(v).strip().lower()]


def parse_age(v) -> int:
    if isinstance(v, int):
        return v if v <= 9 else v // 10
    s = str(v).strip()
    if s in AGE2ID:
        return AGE2ID[s]
    if s.endswith("+"):
        s = s[:-1]
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        decade = int(digits) // 10 if int(digits) >= 10 else int(digits)
        decade = max(1, min(6, decade))   # 60대 이상은 6으로 통일
        return decade
    raise ValueError(f"Cannot parse age: {v}")


class AudioAgeGenderDataset(Dataset):
    def __init__(self, manifest_path: str, sample_rate: int = 16000, max_duration: float = 8.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.items: List[Dict] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            # 간단 선형 리샘플 (속도 우선). 품질이 중요하면 librosa.resample 사용.
            ratio = self.sample_rate / sr
            n_new = int(round(len(wav) * ratio))
            xp = np.arange(len(wav), dtype=np.float32)
            x_new = np.linspace(0, len(wav) - 1, n_new, dtype=np.float32)
            wav = np.interp(x_new, xp, wav).astype(np.float32)
        if len(wav) > self.max_samples:
            wav = wav[: self.max_samples]
        return torch.from_numpy(wav)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        item = self.items[idx]
        try:
            wav = self._load_audio(item["audio"])
        except Exception as e:
            print(f"[load_fail] {item['audio']} :: {e!r}", flush=True)
            # 다음 인덱스로 폴백
            return self.__getitem__((idx + 1) % len(self.items))
        gender = parse_gender(item["gender"])
        age = parse_age(item["age"])
        return wav, gender, age


def collate_fn(batch):
    wavs, genders, ages = zip(*batch)
    lengths = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
    for i, w in enumerate(wavs):
        padded[i, : w.numel()] = w
    return (
        padded,
        lengths,
        torch.tensor(genders, dtype=torch.float32),
        torch.tensor([a - 1 for a in ages], dtype=torch.long),   # 0..5 (class index)
    )


def build_loader(
    manifest: str,
    batch_size: int,
    num_workers: int,
    sample_rate: int,
    max_duration: float,
    shuffle: bool,
) -> DataLoader:
    ds = AudioAgeGenderDataset(manifest, sample_rate, max_duration)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
    )
