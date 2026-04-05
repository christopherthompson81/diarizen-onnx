from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiariZenConfig:
    sample_rate: int = 16_000
    chunk_duration_seconds: int = 16
    segmentation_step: float = 0.1
    frame_rate: int = 50
    num_powerset_classes: int = 11
    max_unique_speakers: int = 4
    max_simultaneous_per_frame: int = 2

    threshold: float = 0.5
    median_filter_size: int = 11
    min_cluster_size: int = 13
    ahc_threshold: float = 0.6
    vbx_fa: float = 0.07
    vbx_fb: float = 0.8
    vbx_max_iters: int = 20
    min_embedding_cluster_size: int = 8
    min_active_frames_for_embed: int = 10
    min_clean_frame_ratio_for_clustering: float = 0.1
    fill_short_gap_frames: int = 2
    min_region_frames: int = 2
    merge_gap_frames: int = 50

    embedding_dim: int = 256
    lda_dim: int = 128
    num_mel_bins: int = 80
    frame_length_samples: int = 400
    frame_shift_samples: int = 160
    fft_size: int = 512
    low_freq_hz: float = 20.0
    high_freq_hz: float = 8_000.0
    preemph_coeff: float = 0.97
    waveform_scale: float = 32_768.0
    max_embed_window_sec: int = 5
