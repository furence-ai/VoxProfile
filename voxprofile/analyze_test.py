"""Per-class confusion / MAE analysis on test set."""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()
    def isatty(self):
        return False
    def __getattr__(self, name):
        return getattr(self.streams[0], name)

from .config import TrainConfig
from .dataset import build_loader, AGE_LABELS
from .model import AgeGenderModel


@torch.no_grad()
def analyze(cfg: TrainConfig, ckpt_path: str, manifest: str, out_png: str | None = None):
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    if out_png is None:
        out_png = os.path.join(ckpt_dir, "confusion.png")
    else:
        out_png = os.path.join(ckpt_dir, os.path.basename(out_png))
    log_path = os.path.join(ckpt_dir, "analyze_test.log")
    log_file = open(log_path, "w")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, log_file)
    try:
        return _analyze_impl(cfg, ckpt_path, manifest, out_png)
    finally:
        sys.stdout = orig_stdout
        log_file.close()


@torch.no_grad()
def _analyze_impl(cfg: TrainConfig, ckpt_path: str, manifest: str, out_png: str):
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

    all_g_true, all_g_pred = [], []
    all_a_true, all_a_round, all_a_pred = [], [], []

    for audio, length, gender, age in loader:
        audio, length = audio.to(device), length.to(device)
        g_logit, a_logits = model(audio, length)
        g_hat = (g_logit.sigmoid() > 0.5).long().cpu()
        a_hat = a_logits.argmax(dim=-1).long().cpu()   # 0..5
        all_g_true.append(gender.long())
        all_g_pred.append(g_hat)
        all_a_true.append(age.long())          # 0..5
        all_a_round.append(a_hat)

    g_true = torch.cat(all_g_true).numpy()
    g_pred = torch.cat(all_g_pred).numpy()
    a_true = torch.cat(all_a_true).numpy() + 1   # → 1..6 for display
    a_round = torch.cat(all_a_round).numpy() + 1
    a_pred_raw = a_round.astype(float)            # no continuous prediction

    n = len(g_true)
    print(f"\n=== Test set: {n} samples ===\n")

    # ---- Gender confusion (2x2) ----
    print("Gender confusion (rows=true, cols=pred)  0=M, 1=F")
    gc = np.zeros((2, 2), dtype=int)
    for t, p in zip(g_true, g_pred):
        gc[t, p] += 1
    print(f"        pred_M  pred_F")
    print(f"true_M  {gc[0,0]:6d}  {gc[0,1]:6d}")
    print(f"true_F  {gc[1,0]:6d}  {gc[1,1]:6d}")
    print(f"Overall gender acc: {(g_true == g_pred).mean():.4f}\n")

    # ---- Age confusion (6x6) ----
    print("Age confusion (rows=true decade, cols=pred decade)")
    ac = np.zeros((6, 6), dtype=int)
    for t, p in zip(a_true, a_round):
        ac[t-1, p-1] += 1
    header = "       " + " ".join(f"{l:>6}" for l in AGE_LABELS)
    print(header)
    for i, lab in enumerate(AGE_LABELS):
        row = " ".join(f"{ac[i,j]:6d}" for j in range(6))
        print(f"{lab:>6} {row}")
    print()

    # ---- Per-class age MAE + decade acc ----
    print("Per-class age stats:")
    print(f"{'class':>7} {'n':>5} {'MAE':>7} {'dec_acc':>8} {'mean_pred':>10}")
    for c in range(1, 7):
        mask = a_true == c
        cnt = mask.sum()
        if cnt == 0:
            continue
        mae = np.abs(a_pred_raw[mask] - c).mean()
        acc = (a_round[mask] == c).mean()
        mp = a_pred_raw[mask].mean()
        print(f"{AGE_LABELS[c-1]:>7} {cnt:5d} {mae:7.3f} {acc:8.3f} {mp:10.3f}")
    print()

    # ---- 12-class joint (gender x age) ----
    print("Joint 12-class accuracy (gender AND decade correct):")
    joint = ((g_true == g_pred) & (a_round == a_true)).mean()
    print(f"  {joint:.4f}\n")

    # ---- Where do gender errors live? ----
    print("Gender errors broken down by true age:")
    print(f"{'age':>7} {'n':>5} {'g_err':>6} {'err_rate':>9}")
    for c in range(1, 7):
        mask = a_true == c
        cnt = mask.sum()
        if cnt == 0:
            continue
        err = (g_true[mask] != g_pred[mask]).sum()
        print(f"{AGE_LABELS[c-1]:>7} {cnt:5d} {err:6d} {err/cnt:9.4f}")

    # ---- Optional heatmap ----
    if out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            # gender
            axes[0].imshow(gc, cmap="Blues")
            axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
            axes[0].set_xticklabels(["Male", "Female"]); axes[0].set_yticklabels(["Male", "Female"])
            axes[0].set_title("Gender confusion")
            for i in range(2):
                for j in range(2):
                    axes[0].text(j, i, gc[i, j], ha="center", va="center",
                                 color="white" if gc[i, j] > gc.max()/2 else "black")
            # age (row-normalized)
            ac_norm = ac / ac.sum(axis=1, keepdims=True).clip(min=1)
            axes[1].imshow(ac_norm, cmap="Blues", vmin=0, vmax=1)
            age_labels_en = ["10s", "20s", "30s", "40s", "50s", "60s+"]
            axes[1].set_xticks(range(6)); axes[1].set_yticks(range(6))
            axes[1].set_xticklabels(age_labels_en); axes[1].set_yticklabels(age_labels_en)
            axes[1].set_title("Age confusion (row-normalized)")
            for i in range(6):
                for j in range(6):
                    axes[1].text(j, i, f"{ac[i,j]}", ha="center", va="center", fontsize=8,
                                 color="white" if ac_norm[i, j] > 0.5 else "black")
            fig.tight_layout()
            fig.savefig(out_png, dpi=120)
            print(f"\nSaved: {out_png}")
        except Exception as e:
            print(f"plot skipped: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = TrainConfig()
    analyze(cfg, args.ckpt, args.manifest or cfg.test_manifest, args.out)
