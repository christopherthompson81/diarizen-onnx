#!/usr/bin/env python3
"""
Minimal end-to-end smoke test for diarizen-onnx.

Generates a short synthetic mono WAV, verifies the expected model files are
present, optionally checks CUDA provider availability, and runs a diarization
pass through the Python API.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from diarizen_onnx.diarizer import DiariZenDiarizer


REQUIRED_FILES = [
    "diarizen_segmentation.onnx",
    "diarizen_segmentation.onnx.data",
    "wespeaker_pyannote_weighted.onnx",
    "plda/mean1.bin",
    "plda/lda.bin",
    "plda/mean2.bin",
    "plda/plda_mu.bin",
    "plda/plda_tr.bin",
    "plda/plda_psi.bin",
]


def generate_test_audio(path: Path, sample_rate: int = 16000, duration_seconds: float = 4.0) -> None:
    num_samples = int(sample_rate * duration_seconds)
    pcm = np.zeros((num_samples,), dtype=np.int16)

    # Add a simple tone burst so the file is not pure silence.
    for start_seconds in (0.5, 2.0):
        start = int(start_seconds * sample_rate)
        end = min(num_samples, start + int(0.8 * sample_rate))
        t = np.arange(end - start, dtype=np.float32) / sample_rate
        tone = 0.2 * np.sin(2.0 * math.pi * 440.0 * t)
        pcm[start:end] = np.clip(tone * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal diarizen-onnx smoke test")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing ONNX and PLDA files")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    missing = [rel for rel in REQUIRED_FILES if not (args.model_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected model files under {args.model_dir}: {missing}")

    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    if args.provider == "cuda" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is not available in this environment.")

    with tempfile.TemporaryDirectory(prefix="diarizen_smoke_") as tmpdir:
        audio_path = Path(tmpdir) / "smoke.wav"
        generate_test_audio(audio_path)

        diarizer = DiariZenDiarizer(args.model_dir, provider=args.provider)
        segments = diarizer.diarize_file(str(audio_path))

    print(f"Smoke test completed with {len(segments)} segment(s).")
    for segment in segments[:5]:
        print(f"{segment.start:8.2f} {segment.end:8.2f} {segment.speaker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
