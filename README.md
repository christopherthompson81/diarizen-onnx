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

For the export workflow, use Python `3.10`. The runtime-only path is more
forgiving, but the export scripts have been validated specifically on Python
`3.10`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

That installs a practical all-in-one environment for:

- downloading the reference ONNX bundle
- exporting the ONNX models from upstream sources
- CPU inference

For GPU inference, switch the ONNX Runtime package in the same venv:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --force-reinstall --no-cache-dir "onnxruntime-gpu>=1.17"
```

If you only want the minimal runtime dependencies instead of the full export environment:

```bash
pip install -r requirements-runtime.txt
```

## Quick Start

The most practical path is to download the already-exported reference ONNX bundle:

```bash
python scripts/download_reference_models.py --output-dir ./models
diarizen-onnx \
  --audio /path/to/audio.wav \
  --model-dir ./models \
  --provider cpu
```

For CUDA inference:

```bash
diarizen-onnx \
  --audio /path/to/audio.wav \
  --model-dir ./models \
  --provider cuda \
  --json
```

Minimal smoke test without a bundled audio sample:

```bash
python scripts/smoke_test.py --model-dir ./models --provider cpu
python scripts/smoke_test.py --model-dir ./models --provider cuda
```

The smoke test generates a short synthetic WAV on the fly instead of shipping a
test clip in the repository. That keeps the repo lightweight and avoids mixing
code distribution with any sample-media licensing questions.

## Models

This repository does not bundle the upstream DiariZen-derived weights.

The MIT license in this repository applies to the source code only. It does not
extend to model weights, checkpoints, or other upstream model artifacts unless a
particular asset explicitly says otherwise.

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

Relevant upstream sources for the weight lineage and licensing context:

- Upstream DiariZen project: `https://github.com/BUTSpeechFIT/DiariZen`
- Upstream DiariZen model repo: `https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md`

Those upstream repositories are the source of the non-commercial licensing
constraints associated with the DiariZen-derived weights. See [`NOTICE`](./NOTICE)
for the code-versus-weights license boundary.

To download the public reference ONNX bundle used by this repository:

```bash
python scripts/download_reference_models.py --output-dir ./models
```

That command downloads the files expected by the runtime layout above.

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
  --provider cpu \
  --json
```

## Export scripts

Scripts are under [`scripts/`](./scripts):

- `download_reference_models.py`
- `export_diarizen_onnx.py`
- `export_pyannote_wespeaker_onnx.py`
- `export_lda_transform.py`

These are intentionally thin wrappers around the upstream DiariZen and pyannote model definitions, so they require a compatible Python environment and access to the upstream checkpoints.

### Export Everything From Upstream Sources

If you want to produce the full runtime model folder yourself:

1. Clone the upstream DiariZen repository with its dependencies:

```bash
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git
```

2. Create and activate a venv, then install the export dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Export the segmentation model:

```bash
python scripts/export_diarizen_onnx.py \
  --diarizen-root ./DiariZen \
  --output-dir ./models
```

4. Export the weighted WeSpeaker embedder used by the runtime:

```bash
python scripts/export_pyannote_wespeaker_onnx.py \
  --diarizen-root ./DiariZen \
  --weighted \
  --output-dir ./models
```

5. Export the LDA/PLDA binaries:

```bash
python scripts/export_lda_transform.py --output-dir ./models/plda
```

After those steps, `./models` should match the expected runtime layout shown above.

You can then run inference directly from the exported model folder:

```bash
diarizen-onnx \
  --audio /path/to/audio.wav \
  --model-dir ./models \
  --provider cuda \
  --json
```

## Validated Matrix

| Workflow | Python | Status |
|---|---|---|
| Reference-weight download | 3.10 | Validated |
| Reference-weight CPU inference | 3.10 | Validated |
| Reference-weight CUDA inference | 3.10 with CUDA-swapped runtime venv | Validated |
| Synthetic smoke test script | 3.10 with CUDA-swapped runtime venv | Validated |
| Export segmentation ONNX | 3.10 | Validated |
| Export weighted WeSpeaker ONNX | 3.10 | Validated |
| Export LDA/PLDA binaries | 3.10 | Validated |
| CUDA inference from freshly exported models | 3.10 export + CUDA-swapped runtime venv | Validated |

The export path has not been validated on Python `3.12`.

## Troubleshooting

- If CUDA inference fails after switching ONNX Runtime packages, run:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install --force-reinstall --no-cache-dir "onnxruntime-gpu>=1.17"
```

- If `python3.10 -m venv` is unavailable on your system, a `virtualenv`-based
  equivalent also works:

```bash
virtualenv -p python3.10 .venv
```

- The export scripts expect a local DiariZen checkout. Pass it explicitly with
  `--diarizen-root /path/to/DiariZen`.

- The weighted WeSpeaker export currently writes a valid runtime model but may
  still emit benign PyTorch/ONNX export warnings during conversion.

## Notes

- Runtime dependencies are intentionally small: `numpy`, `scipy`, `onnxruntime`, `soundfile`.
- Export dependencies are separate because they require PyTorch and Hugging Face tooling.
- `requirements-runtime.txt` is the simplest inference-only setup.
- `requirements.txt` is the practical all-in-one setup for download, export, and CPU inference.
- The clustering path follows the current reference implementation behavior, including constrained per-chunk centroid assignment and the 1.0 s final same-speaker merge gap.

## License Boundary

- The source code in this repository is MIT licensed.
- Model weights are not included in this repository.
- DiariZen-derived weights referenced by this project are separate material with their own license terms.
- The upstream DiariZen weight lineage should be treated as `CC BY-NC 4.0` unless a more specific upstream notice says otherwise.
- If you use, redistribute, host, or ship weights, you are responsible for complying with the applicable upstream model licenses.

## Status

- This repository is intended to be a practical reference implementation of the ONNX-based DiariZen path used here, not a fresh redesign.
- On the standard sample used during development, the current CUDA path is close to the reference output shape and runs in the same rough range, but it is still somewhat slower.
- The exported segmentation ONNX model currently falls back to single-chunk inference at runtime instead of stable multi-chunk batching.
