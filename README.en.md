# VoxProfile

*[한국어 README](README.md)*

Predicts a speaker's **gender** and **age decade** from a short audio clip.
A pretrained speaker-embedding encoder (TitaNet-L) feeds two lightweight
classification heads, so both attributes come out of a single forward pass.

- Gender: binary classification (`Male` / `Female`)
- Age: 6-way decade classification (`10s` `20s` `30s` `40s` `50s` `60s+`)

## Architecture

![Model architecture](assets/model_architecture.png)

1. Raw waveform goes through a pretrained **TitaNet-L** speaker-recognition
   encoder, producing a 192-dim speaker embedding (encoder is frozen by
   default; the top layers can be unfrozen, or adapted with LoRA, instead).
2. The embedding is fed into two independent MLP heads (`FlexHead`), one
   producing a single gender logit and the other 6 age-decade logits. Head
   depth is selected from three presets: `light` (direct linear), `medium`,
   or `heavy`.
3. Both losses are combined and trained jointly:
   `loss = BCEWithLogitsLoss(gender) + λ × CrossEntropyLoss(age)`
   (a 2-phase schedule ramps λ and encoder unfreezing over training — see
   `TrainConfig`)

## Performance

Evaluated on a 624-sample held-out test set, using the `runs/2026_05_21`
checkpoint.

| Metric | Value |
|---|---|
| Gender accuracy | 98.9% (7 misclassifications total) |
| 12-class joint accuracy (gender AND age decade both correct) | 68.8% |

Per-class age breakdown (MAE = decade-index distance between true and
predicted age):

| Age decade | n | MAE | Decade acc |
|---|---:|---:|---:|
| 10s | 97 | 0.825 | 60.8% |
| 20s | 113 | 0.212 | 79.6% |
| 30s | 108 | 0.565 | 47.2% |
| 40s | 97 | 0.567 | 45.4% |
| 50s | 73 | 0.247 | 82.2% |
| 60s+ | 136 | 0.257 | 94.9% |

![Confusion matrices](assets/confusion_matrix.png)

Gender is nearly solved (well under 1% error). Age confusion concentrates
in adjacent decades (30s↔40s, 20s↔30s), while the extremes — 10s and 60s+
— separate cleanly from the rest.

## Training data

Utterances with gender and age labels were drawn from several Korean-language
AIHub corpora (free conversation, meeting recordings, speaker-recognition
audio), cleaned, and split by speaker into train/val/test. The data-prep
pipeline is tightly coupled to internal storage paths and isn't included in
this repository.

## Manifest format (JSONL)

`voxprofile/dataset.py` consumes a JSONL manifest, one utterance per line:

```json
{"audio": "/path/a.wav", "gender": "male",   "age": "30대"}
{"audio": "/path/b.wav", "gender": "female", "age": "20대"}
```

## Setup

```bash
pip install -r requirements.txt
```

`voxprofile/model.py` loads the TitaNet-L encoder via NeMo
(`nemo_toolkit[asr]`), from either a HuggingFace/NGC model name or a local
`.nemo` checkpoint path — set in `TrainConfig.encoder_name`.

## Usage

Run everything from the repo root, as a module (`voxprofile` is a package):

```bash
# Evaluate a checkpoint over a manifest
python -m voxprofile.evaluate --ckpt runs/2026_05_21/best_model.pt --manifest path/to/test.jsonl

# Run inference on a single audio file
python -m voxprofile.evaluate --ckpt runs/2026_05_21/best_model.pt --audio path/to/sample.wav

# Per-class confusion / MAE breakdown on the test set
python -m voxprofile.analyze_test --ckpt runs/2026_05_21/best_model.pt --manifest path/to/test.jsonl
```

Hyperparameters (paths, batch size, LR, LoRA, epochs, etc.) are configured in
`voxprofile/config.py` (`TrainConfig`). Head size is selected via
`TrainConfig.model_size = "light" | "medium" | "heavy"`.

## Project layout

```
voxprofile/
├── config.py         TrainConfig (hyperparams, paths) + MODEL_CONFIGS presets
├── model.py           LoRA wrappers, FlexHead (MLP), AgeGenderModel
├── dataset.py          JSONL manifest Dataset, collate_fn, DataLoader builder
├── evaluate.py          Checkpoint evaluation / single-file inference
└── analyze_test.py       Per-class confusion matrix / MAE analysis

assets/               Diagrams and metric images used in this README
requirements.txt
```

Note: data manifests, model checkpoints, and experiment run outputs
(`runs/`) are not tracked in this repo (see `.gitignore`).

## License

[MIT](LICENSE)
