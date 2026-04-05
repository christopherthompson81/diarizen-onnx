# diarizen-onnx

Minimal Python reference implementation of a DiariZen-style ONNX diarization pipeline, using ONNX Runtime for the neural models and NumPy/SciPy for the algorithmic pieces.

This repository is intended to stay close to the exported ONNX pipeline used here, not to be a full reimplementation of the original DiariZen Python stack.

It includes:

- a standalone diarization CLI
- ONNX export helpers for the segmentation and WeSpeaker models
- a lightweight runtime dependency set
- licensing/NOTICE files that separate MIT-licensed code from upstream model-weight terms

## What it includes

- ONNX Runtime diarization pipeline with:
  - 16 s segmentation chunks, 1.6 s stride
  - powerset decoding and median filtering
  - weighted WeSpeaker ONNX embeddings
  - LDA + PLDA-space clustering
  - short-gap filling, short-region removal, and final same-speaker merge
- Minimal CLI
- Export scripts for the ONNX segmentation model, weighted WeSpeaker embedder, and LDA/PLDA transform
- MIT license for this code, plus a `NOTICE` covering the upstream weights

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU inference:

```bash
pip install -e .[cuda]
```

For the export scripts:

```bash
pip install -e .[export]
```

## Models

This repository does not bundle the upstream DiariZen-derived weights.

Expected model layout:

```text
models/
  diarizen_segmentation.onnx
  diarizen_segmentation.onnx.data
  wespeaker_pyannote_weighted.onnx
  plda/
    mean1.bin
    lda.bin
    mean2.bin
    plda_mu.bin
    plda_tr.bin
    plda_psi.bin
```

If you want the already-exported ONNX bundle used by this repository, the current public reference location is:

`https://huggingface.co/christopherthompson81/diarizen_onnx`

## Run

```bash
diarizen-onnx \
  --audio /path/to/audio.wav \
  --model-dir /path/to/models \
  --provider cpu
```

JSON output:

```bash
diarizen-onnx \
  --audio /path/to/audio.wav \
  --model-dir /path/to/models \
  --json
```

## Export scripts

Scripts are under [`scripts/`](./scripts):

- `export_diarizen_onnx.py`
- `export_pyannote_wespeaker_onnx.py`
- `export_lda_transform.py`

These are intentionally thin wrappers around the upstream DiariZen and pyannote model definitions, so they require a compatible Python environment and access to the upstream checkpoints.

## Notes

- Runtime dependencies are intentionally small: `numpy`, `scipy`, `onnxruntime`, `soundfile`.
- Export dependencies are separate because they require PyTorch and Hugging Face tooling.
- The clustering path follows the current reference implementation behavior, including constrained per-chunk centroid assignment and the 1.0 s final same-speaker merge gap.

## Status

- This repository is intended to be a practical reference implementation of the ONNX-based DiariZen path used here, not a fresh redesign.
- On the standard sample used during development, the current CUDA path is close to the reference output shape and runs in the same rough range, but it is still somewhat slower.
- The exported segmentation ONNX model currently falls back to single-chunk inference at runtime instead of stable multi-chunk batching.
