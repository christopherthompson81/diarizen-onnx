from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import DiariZenConfig


@dataclass(frozen=True)
class PreparedEmbeddingInput:
    fbank: np.ndarray
    frame_weights: np.ndarray | None
    selected_frame_count: int


class WeSpeakerEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        plda_dir: str | Path | None = None,
        provider: str = "cpu",
        config: DiariZenConfig | None = None,
    ) -> None:
        self.config = config or DiariZenConfig()
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=_providers_for(provider),
        )
        self.input_names = [meta.name for meta in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        self.fbank_input_name = self.input_names[0]
        self.weights_input_name = next(
            (name for name in self.input_names if name.lower() == "weights"),
            None,
        )
        self.supports_weights = self.weights_input_name is not None
        providers = self.session.get_providers()
        self.prefer_gpu_batching = provider.lower() != "cpu" and "CUDAExecutionProvider" in providers and self.supports_weights

        self.window = _make_hamming_window(self.config.frame_length_samples)
        self.mel_filters = _make_mel_filters(self.config)

        self.lda_mean1 = None
        self.lda_matrix = None
        self.lda_mean2 = None
        self.plda_mu = None
        self.plda_tr = None
        self.plda_psi = None
        if plda_dir is not None:
            self._load_plda(Path(plda_dir))

    @property
    def has_plda(self) -> bool:
        return self.lda_matrix is not None and self.plda_mu is not None and self.plda_tr is not None

    @property
    def min_num_samples(self) -> int:
        return self.config.frame_length_samples

    def compute_chunk_fbank(self, waveform: np.ndarray) -> np.ndarray:
        return self._compute_fbank(np.asarray(waveform, dtype=np.float32))

    def prepare_embedding_input(
        self,
        fbank: np.ndarray,
        frame_mask: np.ndarray,
    ) -> PreparedEmbeddingInput | None:
        if fbank.shape[0] == 0 or frame_mask.size == 0:
            return None

        frame_mask = np.asarray(frame_mask, dtype=bool)
        num_frames = fbank.shape[0]
        diar_frames = frame_mask.shape[0]
        src_indices = np.floor(np.arange(num_frames) * diar_frames / num_frames).astype(np.int32)
        src_indices = np.clip(src_indices, 0, diar_frames - 1)
        frame_weights = frame_mask[src_indices].astype(np.float32)
        selected = int(frame_weights.sum())
        if selected == 0:
            return None

        if self.supports_weights:
            return PreparedEmbeddingInput(fbank=fbank, frame_weights=frame_weights, selected_frame_count=selected)

        masked = fbank[frame_weights > 0.0]
        return PreparedEmbeddingInput(fbank=masked, frame_weights=None, selected_frame_count=selected)

    def compute_embedding(self, prepared: PreparedEmbeddingInput) -> tuple[np.ndarray, np.ndarray]:
        raw = self._run_inference(prepared.fbank, prepared.frame_weights)
        return _normalize_embedding(raw), raw

    def compute_embeddings_batch(
        self,
        prepared_inputs: list[PreparedEmbeddingInput],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if not prepared_inputs:
            return []

        if not self.supports_weights or any(item.frame_weights is None for item in prepared_inputs):
            return [self.compute_embedding(item) for item in prepared_inputs]

        batch = len(prepared_inputs)
        max_frames = max(item.fbank.shape[0] for item in prepared_inputs)
        data = np.zeros((batch, max_frames, self.config.num_mel_bins), dtype=np.float32)
        weights = np.zeros((batch, max_frames), dtype=np.float32)

        for batch_index, item in enumerate(prepared_inputs):
            num_frames = item.fbank.shape[0]
            data[batch_index, :num_frames, :] = item.fbank
            weights[batch_index, :num_frames] = item.frame_weights

        output = self.session.run(
            [self.output_name],
            {
                self.fbank_input_name: data,
                self.weights_input_name: weights,
            },
        )[0]

        results: list[tuple[np.ndarray, np.ndarray]] = []
        for batch_index in range(batch):
            raw = np.asarray(output[batch_index], dtype=np.float32)
            results.append((_normalize_embedding(raw), raw))
        return results

    def compute_xvec(self, raw_embedding: np.ndarray) -> np.ndarray | None:
        if self.lda_matrix is None:
            return None
        return self._apply_xvec_tf(raw_embedding)

    def apply_plda_tf_batch(self, xvecs: np.ndarray) -> np.ndarray:
        if self.plda_tr is None or self.plda_mu is None:
            raise ValueError("PLDA transform is not available.")
        centered = xvecs - self.plda_mu[None, :]
        return centered @ self.plda_tr.T

    def _load_plda(self, plda_dir: Path) -> None:
        self.lda_mean1 = _read_float_bin(plda_dir / "mean1.bin")
        self.lda_matrix = _read_float_bin(plda_dir / "lda.bin").reshape(self.config.embedding_dim, self.config.lda_dim)
        self.lda_mean2 = _read_float_bin(plda_dir / "mean2.bin")
        self.plda_mu = _read_float_bin(plda_dir / "plda_mu.bin")
        self.plda_tr = _read_float_bin(plda_dir / "plda_tr.bin").reshape(self.config.lda_dim, self.config.lda_dim)
        psi_path = plda_dir / "plda_psi.bin"
        if psi_path.exists():
            self.plda_psi = _read_float_bin(psi_path)

    def _apply_xvec_tf(self, raw_embedding: np.ndarray) -> np.ndarray:
        assert self.lda_mean1 is not None
        assert self.lda_matrix is not None
        assert self.lda_mean2 is not None

        centered = raw_embedding.astype(np.float32) - self.lda_mean1
        centered = _l2_normalize(centered) * float(np.sqrt(centered.shape[0]))
        projected = centered @ self.lda_matrix - self.lda_mean2
        return _l2_normalize(projected) * float(np.sqrt(projected.shape[0]))

    def _compute_fbank(self, waveform: np.ndarray) -> np.ndarray:
        cfg = self.config
        n = int(waveform.shape[0])
        num_frames = 1 + (n - cfg.frame_length_samples) // cfg.frame_shift_samples
        if num_frames <= 0:
            return np.zeros((0, cfg.num_mel_bins), dtype=np.float32)

        frames = np.lib.stride_tricks.sliding_window_view(
            waveform,
            cfg.frame_length_samples,
        )[:: cfg.frame_shift_samples]
        frames = np.asarray(frames[:num_frames], dtype=np.float32).copy()

        frames *= cfg.waveform_scale
        frames -= frames.mean(axis=1, keepdims=True, dtype=np.float32)

        previous = frames[:, :-1].copy()
        frames[:, 1:] -= cfg.preemph_coeff * previous
        frames[:, 0] *= 1.0 - cfg.preemph_coeff

        windowed = frames * self.window[None, :]
        spectrum = np.fft.rfft(windowed, n=cfg.fft_size, axis=1)
        power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float32, copy=False)
        energies = power @ self.mel_filters.T
        result = np.log(np.maximum(energies, np.float32(1e-10)))
        result -= result.mean(axis=0, keepdims=True)
        return result.astype(np.float32, copy=False)

    def _run_inference(self, fbank: np.ndarray, frame_weights: np.ndarray | None) -> np.ndarray:
        fbank = np.asarray(fbank, dtype=np.float32)
        inputs: dict[str, np.ndarray] = {self.fbank_input_name: fbank[None, :, :]}
        if self.supports_weights and frame_weights is not None:
            inputs[self.weights_input_name] = np.asarray(frame_weights, dtype=np.float32)[None, :]
        output = self.session.run([self.output_name], inputs)[0]
        return np.asarray(output[0], dtype=np.float32)


def _providers_for(provider: str) -> list[str]:
    provider = provider.lower()
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _normalize_embedding(raw_embedding: np.ndarray) -> np.ndarray:
    raw_embedding = np.asarray(raw_embedding, dtype=np.float32)
    norm = np.linalg.norm(raw_embedding)
    if norm <= 0:
        return np.zeros_like(raw_embedding)
    return raw_embedding / norm


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= 0:
        return np.zeros_like(x)
    return x / norm


def _read_float_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def _make_hamming_window(size: int) -> np.ndarray:
    return np.hamming(size).astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    return 1127.0 * np.log(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _make_mel_filters(config: DiariZenConfig) -> np.ndarray:
    num_freq_bins = config.fft_size // 2 + 1
    filters = np.zeros((config.num_mel_bins, num_freq_bins), dtype=np.float32)
    mel_low = _hz_to_mel(config.low_freq_hz)
    mel_high = _hz_to_mel(config.high_freq_hz)
    hz_pts = _mel_to_hz(
        np.linspace(mel_low, mel_high, config.num_mel_bins + 2, dtype=np.float64)
    )

    for m in range(config.num_mel_bins):
        fl = hz_pts[m]
        fc = hz_pts[m + 1]
        fh = hz_pts[m + 2]
        for k in range(num_freq_bins):
            fk = k * config.sample_rate / config.fft_size
            if fl <= fk < fc and fc > fl:
                weight = (fk - fl) / (fc - fl)
            elif fc <= fk <= fh and fh > fc:
                weight = (fh - fk) / (fh - fc)
            else:
                weight = 0.0
            filters[m, k] = weight
    return filters
