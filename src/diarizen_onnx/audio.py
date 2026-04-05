from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio_mono(audio_path: str | Path, target_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        audio = resample_poly(audio, target_sample_rate, sample_rate).astype(np.float32, copy=False)

    return np.asarray(audio, dtype=np.float32)
