"""Model: TitaNet encoder + FlexHead for gender & age."""
import math
from typing import List, Tuple

import torch
import torch.nn as nn


# ── LoRA wrappers ───────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Linear + low-rank additive path. base frozen, A/B trainable."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def set_trainable(self, flag: bool) -> None:
        self.A.requires_grad = flag
        self.B.requires_grad = flag

    # NeMo가 base layer attribute에 직접 접근하는 경우 (e.g. forward_for_export)
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


class LoRAConv1d1x1(nn.Module):
    """1x1 Conv1d + low-rank additive path. Treats channels like Linear features."""

    def __init__(self, base: nn.Conv1d, rank: int, alpha: int):
        super().__init__()
        assert base.kernel_size == (1,), "LoRAConv1d1x1 requires kernel_size=1"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Conv1d(base.in_channels, rank, kernel_size=1, bias=False)
        self.B = nn.Conv1d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * self.scale

    def set_trainable(self, flag: bool) -> None:
        for p in self.A.parameters():
            p.requires_grad = flag
        for p in self.B.parameters():
            p.requires_grad = flag

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_channels(self):
        return self.base.in_channels

    @property
    def out_channels(self):
        return self.base.out_channels

    @property
    def kernel_size(self):
        return self.base.kernel_size


def _apply_lora_recursive(module: nn.Module, rank: int, alpha: int) -> int:
    """Replace Linear & 1x1 Conv1d in module with LoRA wrappers. Returns # wrapped."""
    n_wrapped = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRALinear, LoRAConv1d1x1)):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            n_wrapped += 1
        elif isinstance(child, nn.Conv1d) and child.kernel_size == (1,):
            setattr(module, name, LoRAConv1d1x1(child, rank, alpha))
            n_wrapped += 1
        else:
            n_wrapped += _apply_lora_recursive(child, rank, alpha)
    return n_wrapped


# ── Head ────────────────────────────────────────────────────────────────
class FlexHead(nn.Module):
    """Configurable MLP head."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full model ──────────────────────────────────────────────────────────
class AgeGenderModel(nn.Module):
    """TitaNet encoder + dual heads (gender, age decade)."""

    def __init__(
        self,
        encoder_name: str,
        encoder_out_dim: int,
        hidden_dims: List[int],
        dropout: float,
        unfreeze_encoder_layers: int = 0,
        num_age_classes: int = 6,
        emb_dropout: float = 0.0,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.encoder = self._load_titanet(encoder_name)
        self.encoder_out_dim = encoder_out_dim
        self.num_age_classes = num_age_classes
        self.use_lora = use_lora

        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        self.gender_head = FlexHead(encoder_out_dim, 1, hidden_dims, dropout)
        self.age_head = FlexHead(encoder_out_dim, num_age_classes, hidden_dims, dropout)

        self.freeze_encoder()
        if use_lora:
            n = _apply_lora_recursive(self.encoder, lora_rank, lora_alpha)
            print(f"[LoRA] wrapped {n} layers (rank={lora_rank}, alpha={lora_alpha})")
            # phase 1: LoRA frozen too — head만 학습
            self.set_lora_trainable(False)
        elif unfreeze_encoder_layers > 0:
            self.unfreeze_last_n(unfreeze_encoder_layers)

    @staticmethod
    def _load_titanet(name_or_path: str) -> nn.Module:
        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel
        except ImportError as e:
            raise ImportError(
                "nemo_toolkit[asr] required: pip install nemo_toolkit[asr]"
            ) from e
        import os
        if os.path.isfile(name_or_path) and name_or_path.endswith(".nemo"):
            return EncDecSpeakerLabelModel.restore_from(restore_path=name_or_path)
        return EncDecSpeakerLabelModel.from_pretrained(model_name=name_or_path)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_n(self, n: int) -> None:
        layers = None
        for path in ("encoder.encoder.layers", "encoder.layers"):
            obj = self
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, nn.Sequential)):
                    layers = obj
                    break
            except AttributeError:
                continue
        if layers is None:
            for p in self.encoder.parameters():
                p.requires_grad = True
            return
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def set_lora_trainable(self, flag: bool) -> None:
        """LoRA A/B 파라미터들의 requires_grad 토글."""
        for m in self.encoder.modules():
            if isinstance(m, (LoRALinear, LoRAConv1d1x1)):
                m.set_trainable(flag)

    def encode(self, audio: torch.Tensor, audio_len: torch.Tensor) -> torch.Tensor:
        _, emb = self.encoder.forward(input_signal=audio, input_signal_length=audio_len)
        return emb

    def forward(
        self, audio: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(audio, audio_len)
        emb = self.emb_dropout(emb)
        gender_logit = self.gender_head(emb).squeeze(-1)
        age_logits = self.age_head(emb)
        return gender_logit, age_logits

    def trainable_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        head_params, enc_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("encoder."):
                enc_params.append(p)
            else:
                head_params.append(p)
        return head_params, enc_params
