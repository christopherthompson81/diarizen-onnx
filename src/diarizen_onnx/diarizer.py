from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from math import log
import os

import numpy as np
import onnxruntime as ort

from .audio import load_audio_mono
from .config import DiariZenConfig
from .embedder import WeSpeakerEmbedder


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class DiariZenDiarizer:
    def __init__(
        self,
        model_dir: str | Path,
        provider: str = "cpu",
        config: DiariZenConfig | None = None,
    ) -> None:
        self.config = config or DiariZenConfig()
        self.model_dir = Path(model_dir)
        self.provider = provider

        self.segmentation_model_path = self.model_dir / "diarizen_segmentation.onnx"
        self.embedder_model_path = self.model_dir / "wespeaker_pyannote_weighted.onnx"
        self.plda_dir = self.model_dir / "plda"

        self.seg_sessions = [
            ort.InferenceSession(
                str(self.segmentation_model_path),
                providers=_providers_for(provider),
                sess_options=_make_segmentation_session_options(provider),
            )
            for _ in range(_determine_segmentation_worker_count(provider))
        ]
        self.seg_session = self.seg_sessions[0]
        self.seg_input_name = self.seg_session.get_inputs()[0].name
        self.seg_output_name = self.seg_session.get_outputs()[0].name
        self.seg_supports_batching = _supports_dynamic_batch(self.seg_session)
        self.seg_batch_size = _default_segmentation_batch_size(provider, self.seg_supports_batching)
        self.seg_batching_disabled = False
        self.embedder = WeSpeakerEmbedder(
            self.embedder_model_path,
            plda_dir=self.plda_dir,
            provider=provider,
            config=self.config,
        )
        self.powerset_combinations = _powerset_combinations(
            self.config.max_unique_speakers,
            self.config.max_simultaneous_per_frame,
        )

    def diarize_file(
        self,
        audio_path: str | Path,
        min_speakers: int = 1,
        max_speakers: int = 8,
        threshold: float | None = None,
        ahc_threshold: float | None = None,
    ) -> list[DiarizationSegment]:
        audio = load_audio_mono(audio_path, self.config.sample_rate)
        return self.diarize(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            threshold=threshold,
            ahc_threshold=ahc_threshold,
        )

    def diarize(
        self,
        audio: np.ndarray,
        min_speakers: int = 1,
        max_speakers: int = 8,
        threshold: float | None = None,
        ahc_threshold: float | None = None,
    ) -> list[DiarizationSegment]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return []

        threshold = self.config.threshold if threshold is None else threshold
        ahc_threshold = self.config.ahc_threshold if ahc_threshold is None else ahc_threshold

        start_times: list[float] = []
        per_chunk_binary: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        xvecs: list[np.ndarray] = []
        results: list[EmbeddingJobResult] = []
        prepared_jobs: list[PreparedJob] = []
        prep_max_workers = max(1, min(4, max(1, _cpu_count() - 2)))
        prep_futures: list[Future[list[PreparedJob]]] = []

        with ThreadPoolExecutor(max_workers=prep_max_workers) as prep_executor:
            for chunk_batch in self._enumerate_chunk_batches(audio, self.seg_batch_size):
                raw_batch_scores = self._segment_batch([chunk for _idx, chunk, _start in chunk_batch])
                for (chunk_idx, chunk, start_time), raw_scores in zip(chunk_batch, raw_batch_scores, strict=True):
                    chunk_score = self._decode_chunk_binary(raw_scores, threshold)
                    per_chunk_binary.append(chunk_score)
                    start_times.append(start_time)
                    prep_futures.append(prep_executor.submit(self._prepare_chunk_jobs, chunk_idx, chunk, chunk_score))

            for future in prep_futures:
                prepared_jobs.extend(future.result())

        if self.embedder.prefer_gpu_batching:
            for batch in _make_embedding_batches(
                prepared_jobs,
                max_batch_size=32,
                max_batch_frames=32_000,
            ):
                batch_outputs = self.embedder.compute_embeddings_batch([job.prepared for job in batch])
                for job, (l2_embedding, raw_embedding) in zip(batch, batch_outputs, strict=True):
                    xvec = self.embedder.compute_xvec(raw_embedding)
                    embeddings.append(l2_embedding.astype(np.float64))
                    if xvec is not None:
                        xvecs.append(xvec.astype(np.float32))
                    results.append(
                        EmbeddingJobResult(
                            chunk_idx=job.chunk_idx,
                            speaker_idx=job.speaker_idx,
                            num_frames=job.num_frames,
                            clean_frame_count=job.clean_frame_count,
                        )
                    )
        else:
            for job in prepared_jobs:
                l2_embedding, raw_embedding = self.embedder.compute_embedding(job.prepared)
                xvec = self.embedder.compute_xvec(raw_embedding)
                embeddings.append(l2_embedding.astype(np.float64))
                if xvec is not None:
                    xvecs.append(xvec.astype(np.float32))
                results.append(
                    EmbeddingJobResult(
                        chunk_idx=job.chunk_idx,
                        speaker_idx=job.speaker_idx,
                        num_frames=job.num_frames,
                        clean_frame_count=job.clean_frame_count,
                    )
                )

        if not embeddings:
            return []

        embedding_matrix = np.asarray(embeddings, dtype=np.float64)
        train_indices = _select_training_embedding_indices(
            results,
            self.config.min_clean_frame_ratio_for_clustering,
        )
        train_embeddings = embedding_matrix[train_indices]

        if self.embedder.has_plda and len(results) == len(xvecs):
            fea_batch = self.embedder.apply_plda_tf_batch(np.asarray(xvecs, dtype=np.float32))
            train_features = np.asarray(fea_batch[train_indices], dtype=np.float64)
        else:
            train_features = train_embeddings

        cluster_ids = _cluster_embeddings(train_embeddings, threshold=ahc_threshold)

        if self.embedder.has_plda and self.embedder.plda_psi is not None and len(results) == len(xvecs):
            train_cluster_ids = _vbx_cluster(
                train_features,
                np.asarray(self.embedder.plda_psi, dtype=np.float64),
                cluster_ids,
                fa=self.config.vbx_fa,
                fb=self.config.vbx_fb,
                max_iters=self.config.vbx_max_iters,
            )
            train_cluster_ids = _merge_small_clusters(
                train_cluster_ids,
                train_embeddings,
                self.config.min_embedding_cluster_size,
            )
            centroids = _compute_cluster_centroids(train_embeddings, train_cluster_ids)
            cluster_ids = _assign_embeddings_to_centroids(
                results,
                embedding_matrix,
                centroids,
                constrained=True,
            )
        else:
            train_cluster_ids = _merge_small_clusters(cluster_ids, train_embeddings, self.config.min_cluster_size)
            centroids = _compute_cluster_centroids(train_embeddings, train_cluster_ids)
            cluster_ids = _assign_embeddings_to_centroids(
                results,
                embedding_matrix,
                centroids,
                constrained=False,
            )

        local_to_global: dict[tuple[int, int], int] = {}
        for index, result in enumerate(results):
            if cluster_ids[index] < 0:
                continue
            local_to_global[(result.chunk_idx, result.speaker_idx)] = int(cluster_ids[index])

        num_global_speakers = (
            max((speaker_id for speaker_id in local_to_global.values() if speaker_id >= 0), default=0) + 1
        )

        return self._reconstruct_timeline(
            per_chunk_binary,
            start_times,
            local_to_global,
            num_global_speakers,
            max_speakers,
            int(np.ceil(audio.shape[0] / self.config.sample_rate * self.config.frame_rate)),
        )

    def segments_to_json(self, segments: list[DiarizationSegment]) -> str:
        return json.dumps([asdict(segment) for segment in segments], indent=2)

    def _enumerate_chunks(self, audio: np.ndarray):
        chunk_samples = self.config.chunk_duration_seconds * self.config.sample_rate
        stride_samples = int(chunk_samples * self.config.segmentation_step)
        chunk_index = 0
        for start in range(0, audio.shape[0], stride_samples):
            chunk = np.zeros((chunk_samples,), dtype=np.float32)
            end = min(start + chunk_samples, audio.shape[0])
            chunk[: end - start] = audio[start:end]
            yield chunk_index, chunk, start / self.config.sample_rate
            chunk_index += 1

    def _enumerate_chunk_batches(self, audio: np.ndarray, batch_size: int):
        batch_size = max(1, batch_size)
        batch: list[tuple[int, np.ndarray, float]] = []
        for chunk_info in self._enumerate_chunks(audio):
            batch.append(chunk_info)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _segment_chunk(self, chunk: np.ndarray) -> np.ndarray:
        inputs = {self.seg_input_name: chunk[None, None, :].astype(np.float32)}
        output = self.seg_session.run([self.seg_output_name], inputs)[0]
        return np.asarray(output[0], dtype=np.float32)

    def _segment_batch(self, chunks: list[np.ndarray]) -> list[np.ndarray]:
        if not chunks:
            return []
        if len(chunks) == 1 or not self.seg_supports_batching or self.seg_batching_disabled:
            if len(chunks) == 1 or len(self.seg_sessions) == 1:
                return [self._segment_chunk(chunk) for chunk in chunks]
            with ThreadPoolExecutor(max_workers=len(self.seg_sessions)) as executor:
                futures = [
                    executor.submit(self._segment_chunk_with_session, self.seg_sessions[index % len(self.seg_sessions)], chunk)
                    for index, chunk in enumerate(chunks)
                ]
                return [future.result() for future in futures]

        try:
            batch_input = np.stack(chunks, axis=0).astype(np.float32, copy=False)[:, None, :]
            output = self.seg_session.run([self.seg_output_name], {self.seg_input_name: batch_input})[0]
            return [np.asarray(output[batch_index], dtype=np.float32) for batch_index in range(output.shape[0])]
        except Exception:
            self.seg_batching_disabled = True
            self.seg_batch_size = 1
            return [self._segment_chunk(chunk) for chunk in chunks]

    def _segment_chunk_with_session(self, session: ort.InferenceSession, chunk: np.ndarray) -> np.ndarray:
        inputs = {self.seg_input_name: chunk[None, None, :].astype(np.float32)}
        output = session.run([self.seg_output_name], inputs)[0]
        return np.asarray(output[0], dtype=np.float32)

    def _prepare_chunk_jobs(
        self,
        chunk_idx: int,
        chunk: np.ndarray,
        chunk_score: np.ndarray,
    ) -> list[PreparedJob]:
        overlap_counts = (chunk_score > 0).sum(axis=1)
        num_frames = chunk_score.shape[0]
        fbank = self.embedder.compute_chunk_fbank(chunk)
        prepared_jobs: list[PreparedJob] = []

        for spk in range(self.config.max_unique_speakers):
            active_mask = chunk_score[:, spk] > 0
            active_count = int(active_mask.sum())
            if active_count == 0:
                continue

            clean_mask = active_mask & (overlap_counts == 1)
            clean_count = int(clean_mask.sum())

            if self.embedder.supports_weights:
                min_mask_frames = max(
                    1,
                    int(np.ceil(num_frames * (self.embedder.min_num_samples / chunk.shape[0]))),
                )
                selected_mask = clean_mask if clean_count > min_mask_frames else active_mask
            else:
                if active_count < self.config.min_active_frames_for_embed:
                    continue
                selected_mask = clean_mask if clean_count >= self.config.min_active_frames_for_embed else active_mask

            prepared = self.embedder.prepare_embedding_input(fbank, selected_mask)
            if prepared is None:
                continue

            prepared_jobs.append(
                PreparedJob(
                    chunk_idx=chunk_idx,
                    speaker_idx=spk,
                    num_frames=num_frames,
                    clean_frame_count=clean_count,
                    prepared=prepared,
                )
            )

        return prepared_jobs

    def _decode_chunk_binary(self, logits: np.ndarray, threshold: float) -> np.ndarray:
        probs = _softmax(logits, axis=1)
        num_frames = probs.shape[0]
        scores = np.zeros((num_frames, self.config.max_unique_speakers), dtype=np.float32)
        window_size = self.config.median_filter_size
        if window_size <= 1:
            _accumulate_speaker_scores(probs, scores, self.powerset_combinations)
            return (scores >= threshold).astype(np.float32)

        half = window_size // 2
        for class_index, speakers in enumerate(self.powerset_combinations):
            if not speakers:
                continue
            series = probs[:, class_index]
            padded = np.pad(series, (half, half), mode="reflect")
            windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
            medians = np.median(windows, axis=1).astype(np.float32, copy=False)
            for speaker in speakers:
                scores[:, speaker] += medians
        return (scores >= threshold).astype(np.float32)

    def _reconstruct_timeline(
        self,
        per_chunk_binary: list[np.ndarray],
        start_times: list[float],
        local_to_global: dict[tuple[int, int], int],
        num_global_speakers: int,
        max_speakers: int,
        total_frames: int,
    ) -> list[DiarizationSegment]:
        activation_sums = np.zeros((total_frames, num_global_speakers), dtype=np.float32)
        activation_contrib = np.zeros((total_frames, num_global_speakers), dtype=np.int32)
        count_sums = np.zeros((total_frames,), dtype=np.float32)
        count_contrib = np.zeros((total_frames,), dtype=np.int32)
        chunk_has_global = np.zeros((len(per_chunk_binary), num_global_speakers), dtype=bool)
        start_frames = np.rint(np.asarray(start_times) * self.config.frame_rate).astype(np.int32)

        for (chunk_idx, _local_speaker), global_speaker in local_to_global.items():
            if global_speaker >= 0:
                chunk_has_global[chunk_idx, global_speaker] = True

        for ci, chunk_score in enumerate(per_chunk_binary):
            num_frames = chunk_score.shape[0]
            start_frame = int(start_frames[ci])
            end_frame = min(total_frames, start_frame + num_frames)
            valid_frames = end_frame - start_frame
            if valid_frames <= 0:
                continue

            chunk_slice = chunk_score[:valid_frames]
            chunk_global = np.zeros((valid_frames, num_global_speakers), dtype=np.float32)
            for spk in range(self.config.max_unique_speakers):
                gs = local_to_global.get((ci, spk))
                if gs is None or gs < 0:
                    continue
                chunk_global[:, gs] = np.maximum(chunk_global[:, gs], chunk_slice[:, spk])

            active_mask = chunk_has_global[ci]
            frame_slice = slice(start_frame, end_frame)
            activation_sums[frame_slice] += chunk_global
            activation_contrib[frame_slice] += active_mask[np.newaxis, :].astype(np.int32)
            count_sums[frame_slice] += (chunk_slice > 0).sum(axis=1, dtype=np.int32)
            count_contrib[frame_slice] += 1

        binary = np.zeros((total_frames, num_global_speakers), dtype=bool)
        debug_frames = _create_reconstruction_debug_frames(total_frames)
        for frame_index in range(total_frames):
            if count_contrib[frame_index] <= 0:
                continue
            average_count = count_sums[frame_index] / count_contrib[frame_index]
            count = int(np.floor(average_count + 0.5))
            count = max(0, min(count, min(max_speakers, num_global_speakers)))
            if count == 0:
                continue
            scores = np.zeros((num_global_speakers,), dtype=np.float32)
            contrib_mask = activation_contrib[frame_index] > 0
            scores[contrib_mask] = (
                activation_sums[frame_index, contrib_mask] / activation_contrib[frame_index, contrib_mask]
            )
            # Match the C# stable descending sort: higher score first, lower speaker index first on ties.
            ranked = np.lexsort((np.arange(num_global_speakers, dtype=np.int32), -scores))
            for rank in ranked[:count]:
                if scores[rank] <= 0:
                    break
                binary[frame_index, rank] = True
            _capture_reconstruction_debug_frame(
                debug_frames,
                frame_index,
                float(average_count),
                count,
                scores,
                binary[frame_index],
            )

        _smooth_binary_timeline(
            binary,
            fill_short_gap_frames=self.config.fill_short_gap_frames,
            min_region_frames=self.config.min_region_frames,
        )

        labeled: list[tuple[int, int, str]] = []
        for speaker_index in range(num_global_speakers):
            region_start = None
            for frame_index in range(total_frames):
                active = bool(binary[frame_index, speaker_index])
                if active and region_start is None:
                    region_start = frame_index
                elif not active and region_start is not None:
                    labeled.append((region_start, frame_index, f"speaker_{speaker_index}"))
                    region_start = None
            if region_start is not None:
                labeled.append((region_start, total_frames, f"speaker_{speaker_index}"))

        _write_reconstruction_debug_frames(debug_frames)
        return _merge_adjacent_segments(labeled, self.config.frame_rate, self.config.merge_gap_frames)


@dataclass(frozen=True)
class EmbeddingJobResult:
    chunk_idx: int
    speaker_idx: int
    num_frames: int
    clean_frame_count: int


@dataclass(frozen=True)
class PreparedJob:
    chunk_idx: int
    speaker_idx: int
    num_frames: int
    clean_frame_count: int
    prepared: object


def _providers_for(provider: str) -> list[str]:
    if provider.lower() == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _supports_dynamic_batch(session: ort.InferenceSession) -> bool:
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if not shape:
        return False
    batch_dim = shape[0]
    return batch_dim is None or (isinstance(batch_dim, str)) or (isinstance(batch_dim, int) and batch_dim <= 0)


def _default_segmentation_batch_size(provider: str, supports_batching: bool) -> int:
    if not supports_batching:
        return 1
    return 1

def _make_segmentation_session_options(provider: str) -> ort.SessionOptions:
    options = ort.SessionOptions()
    if provider.lower() == "cpu":
        options.intra_op_num_threads = max(1, min(12, int(np.floor(_cpu_count() * 0.75))))
    return options


def _determine_segmentation_worker_count(provider: str) -> int:
    if provider.lower() != "cpu":
        return 1
    logical = _cpu_count()
    intra = max(1, min(12, int(np.floor(logical * 0.75))))
    return max(1, min(4, logical // intra))


def _cpu_count() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def _powerset_combinations(max_unique_speakers: int, max_simultaneous_per_frame: int) -> list[tuple[int, ...]]:
    combos = [()]
    for size in range(1, max_simultaneous_per_frame + 1):
        combos.extend(combinations(range(max_unique_speakers), size))
    return list(combos)


def _accumulate_speaker_scores(
    probs: np.ndarray,
    scores: np.ndarray,
    powerset_combinations: list[tuple[int, ...]],
) -> None:
    for class_index, speakers in enumerate(powerset_combinations):
        if not speakers:
            continue
        for speaker in speakers:
            scores[:, speaker] += probs[:, class_index]


def _cluster_embeddings(
    embeddings: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if embeddings.shape[0] == 1:
        return np.array([0], dtype=np.int32)
    linkage = _hac_linkage(embeddings)
    return _hac_fcluster_threshold(linkage, threshold)


def _hac_linkage(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    if n < 2:
        return np.zeros((0, 4), dtype=np.float64)

    d = features.shape[1]
    max_clusters = 2 * n - 1
    r = np.full((max_clusters, max_clusters), np.inf, dtype=np.float64)

    base = features.astype(np.float64, copy=False)
    sq = np.sum(base * base, axis=1, keepdims=True)
    initial = np.sqrt(np.maximum(sq + sq.T - 2.0 * (base @ base.T), 0.0))
    r[:n, :n] = initial
    np.fill_diagonal(r[:n, :n], np.inf)

    cluster_size = np.zeros((max_clusters,), dtype=np.int32)
    cluster_size[:n] = 1
    cluster_centroid = np.zeros((max_clusters, d), dtype=np.float64)
    cluster_centroid[:n, :] = base
    active = np.zeros((max_clusters,), dtype=bool)
    active[:n] = True

    linkage = np.zeros((n - 1, 4), dtype=np.float64)
    for k in range(n - 1):
        current_indices = np.flatnonzero(active[: n + k])
        if current_indices.size < 2:
            break
        sub = r[np.ix_(current_indices, current_indices)]
        upper_mask = np.triu(np.ones(sub.shape, dtype=bool), 1)
        masked = np.where(upper_mask, sub, np.inf)
        flat_index = int(np.argmin(masked))
        min_dist = float(masked.flat[flat_index])
        if not np.isfinite(min_dist):
            break
        row, col = np.unravel_index(flat_index, masked.shape)
        i_min = int(current_indices[row])
        j_min = int(current_indices[col])

        linkage[k] = np.array([i_min, j_min, min_dist, cluster_size[i_min] + cluster_size[j_min]], dtype=np.float64)
        new_index = n + k
        cluster_size[new_index] = cluster_size[i_min] + cluster_size[j_min]
        centroid = (
            cluster_size[i_min] * cluster_centroid[i_min] + cluster_size[j_min] * cluster_centroid[j_min]
        ) / cluster_size[new_index]
        cluster_centroid[new_index] = centroid

        active[i_min] = False
        active[j_min] = False
        active[new_index] = True

        other_indices = np.flatnonzero(active[: new_index])
        if other_indices.size > 0:
            diffs = cluster_centroid[other_indices] - centroid
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            r[other_indices, new_index] = dists
            r[new_index, other_indices] = dists

        r[i_min, :] = np.inf
        r[:, i_min] = np.inf
        r[j_min, :] = np.inf
        r[:, j_min] = np.inf
        r[i_min, new_index] = np.inf
        r[new_index, i_min] = np.inf
        r[j_min, new_index] = np.inf
        r[new_index, j_min] = np.inf
        r[new_index, new_index] = np.inf

    return linkage


def _hac_fcluster_threshold(linkage: np.ndarray, threshold: float) -> np.ndarray:
    if linkage.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    n = linkage.shape[0] + 1
    parent = np.full((2 * n - 1,), -1, dtype=np.int32)
    for k, row in enumerate(linkage):
        i = int(row[0])
        j = int(row[1])
        dist = float(row[2])
        new_idx = n + k
        if dist > threshold:
            continue
        parent[i] = new_idx
        parent[j] = new_idx

    labels = np.zeros((n,), dtype=np.int32)
    for i in range(n):
        root = i
        while parent[root] != -1:
            root = int(parent[root])
        labels[i] = root

    mapping: dict[int, int] = {}
    next_label = 0
    for i, root in enumerate(labels):
        key = int(root)
        if key not in mapping:
            mapping[key] = next_label
            next_label += 1
        labels[i] = mapping[key]
    return labels.astype(np.int32)


def _merge_small_clusters(cluster_ids: np.ndarray, embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    values, counts = np.unique(cluster_ids, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts, strict=True)}
    large_set = {cid for cid, count in count_map.items() if count >= min_cluster_size}
    small_set = {cid for cid, count in count_map.items() if count < min_cluster_size}
    if not small_set or not large_set:
        return _compact_ids(cluster_ids)

    centroids = {}
    for cluster_id in large_set:
        centroids[cluster_id] = embeddings[cluster_ids == cluster_id].mean(axis=0)

    result = cluster_ids.copy()
    for index, cluster_id in enumerate(result):
        if int(cluster_id) not in small_set:
            continue
        best_id = min(
            large_set,
            key=lambda candidate: float(np.sum((embeddings[index] - centroids[candidate]) ** 2)),
        )
        result[index] = best_id
    return _compact_ids(result)


def _compact_ids(ids: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    next_id = 0
    result = np.empty_like(ids)
    for index, value in enumerate(ids):
        key = int(value)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        result[index] = mapping[key]
    return result.astype(np.int32)


def _select_training_embedding_indices(
    results: list[EmbeddingJobResult],
    min_clean_frame_ratio: float,
) -> np.ndarray:
    if len(results) <= 1:
        return np.arange(len(results), dtype=np.int32)
    selected = []
    for index, result in enumerate(results):
        min_clean_frames = max(1, int(round(result.num_frames * min_clean_frame_ratio)))
        if result.clean_frame_count >= min_clean_frames:
            selected.append(index)
    if len(selected) >= 2:
        return np.asarray(selected, dtype=np.int32)
    return np.arange(len(results), dtype=np.int32)


def _compute_cluster_centroids(embeddings: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    num_clusters = int(cluster_ids.max()) + 1
    centroids = np.zeros((num_clusters, embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros((num_clusters,), dtype=np.int32)
    for embedding, cluster_id in zip(embeddings, cluster_ids, strict=True):
        centroids[int(cluster_id)] += embedding
        counts[int(cluster_id)] += 1
    for cluster_id in range(num_clusters):
        if counts[cluster_id] > 0:
            centroids[cluster_id] /= counts[cluster_id]
    return centroids


def _assign_embeddings_to_centroids(
    results: list[EmbeddingJobResult],
    embeddings: np.ndarray,
    centroids: np.ndarray,
    constrained: bool,
) -> np.ndarray:
    if not results or centroids.shape[0] == 0:
        return np.zeros((len(results),), dtype=np.int32)

    assigned = np.full((len(results),), -1, dtype=np.int32)
    by_chunk: dict[int, list[int]] = {}
    for index, result in enumerate(results):
        by_chunk.setdefault(result.chunk_idx, []).append(index)

    for chunk_indices in by_chunk.values():
        if not constrained or len(chunk_indices) == 1 or centroids.shape[0] == 1:
            for index in chunk_indices:
                assigned[index] = _best_centroid_index(embeddings[index], centroids)
            continue

        score_matrix = np.zeros((len(chunk_indices), centroids.shape[0]), dtype=np.float64)
        for row, index in enumerate(chunk_indices):
            for col in range(centroids.shape[0]):
                score_matrix[row, col] = _cosine_similarity(embeddings[index], centroids[col])
        row_to_cluster = _best_unique_assignment(score_matrix)
        for row, cluster in enumerate(row_to_cluster):
            if cluster >= 0:
                assigned[chunk_indices[row]] = cluster

    active = assigned[assigned >= 0]
    if active.size == 0:
        return np.zeros_like(assigned)

    compact_map: dict[int, int] = {}
    next_id = 0
    for index, value in enumerate(assigned):
        if value < 0:
            continue
        key = int(value)
        if key not in compact_map:
            compact_map[key] = next_id
            next_id += 1
        assigned[index] = compact_map[key]
    return assigned


def _best_centroid_index(embedding: np.ndarray, centroids: np.ndarray) -> int:
    scores = np.array([_cosine_similarity(embedding, centroid) for centroid in centroids], dtype=np.float64)
    return int(np.argmax(scores))


def _best_unique_assignment(score_matrix: np.ndarray) -> np.ndarray:
    rows, cols = score_matrix.shape
    assignment = np.full((rows,), -1, dtype=np.int32)
    best = np.full((rows,), -1, dtype=np.int32)
    used_cols = np.zeros((cols,), dtype=bool)
    best_score = -np.inf
    target_assignments = min(rows, cols)

    def search(row: int, assigned_count: int, total_score: float) -> None:
        nonlocal best_score
        if row == rows:
            if assigned_count == target_assignments and total_score > best_score:
                best_score = total_score
                best[:] = assignment
            return

        if rows - row < target_assignments - assigned_count:
            return

        if assigned_count < target_assignments:
            for col in range(cols):
                if used_cols[col]:
                    continue
                used_cols[col] = True
                assignment[row] = col
                search(row + 1, assigned_count + 1, total_score + float(score_matrix[row, col]))
                assignment[row] = -1
                used_cols[col] = False

        if assigned_count + (rows - row - 1) >= target_assignments:
            search(row + 1, assigned_count, total_score)

    search(0, 0, 0.0)
    return best


def _make_embedding_batches(
    jobs: list[PreparedJob],
    max_batch_size: int,
    max_batch_frames: int,
) -> list[list[PreparedJob]]:
    batches: list[list[PreparedJob]] = []
    current: list[PreparedJob] = []
    current_max_frames = 0
    for job in jobs:
        frames = job.prepared.fbank.shape[0]
        projected_max = max(current_max_frames, frames)
        projected_frames = (len(current) + 1) * projected_max
        if current and (len(current) >= max_batch_size or projected_frames > max_batch_frames):
            batches.append(current)
            current = []
            current_max_frames = 0
        current.append(job)
        current_max_frames = max(current_max_frames, frames)
    if current:
        batches.append(current)
    return batches


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= 0 or norm_b <= 0:
        return float("-inf")
    return float(np.dot(a, b) / (norm_a * norm_b))


def _vbx_cluster(
    fea: np.ndarray,
    phi: np.ndarray,
    init_labels: np.ndarray,
    fa: float,
    fb: float,
    max_iters: int,
) -> np.ndarray:
    n, d = fea.shape
    k = int(init_labels.max()) + 1
    if n == 0 or k <= 1:
        return init_labels.astype(np.int32)

    smoothing = 7.0
    exp_smoothing = np.exp(smoothing)
    z = exp_smoothing + (k - 1)
    p_correct = exp_smoothing / z
    p_other = 1.0 / z
    gamma = np.full((n, k), p_other, dtype=np.float64)
    gamma[np.arange(n), init_labels.astype(np.int32)] = p_correct

    pi = np.full((k,), 1.0 / k, dtype=np.float64)
    g = -0.5 * (np.sum(fea * fea, axis=1) + d * log(2 * np.pi))
    v = np.sqrt(phi)
    rho = fea * v[None, :]
    prev_elbo = -np.inf
    eps = 1e-8
    epsilon = 1e-4

    for _ in range(max_iters):
        gamma_sum = gamma.sum(axis=0)
        inv_l = 1.0 / (1.0 + (fa / fb) * gamma_sum[:, None] * phi[None, :])
        alpha = ((fa / fb) * inv_l) * (gamma.T @ rho)
        dot = rho @ alpha.T
        reg = ((inv_l + alpha * alpha) * phi[None, :]).sum(axis=1)
        log_p = fa * (dot - 0.5 * reg[None, :] + g[:, None])

        lpi = np.log(pi + eps)
        joint = log_p + lpi[None, :]
        max_lp = joint.max(axis=1, keepdims=True)
        log_px = max_lp + np.log(np.exp(joint - max_lp).sum(axis=1, keepdims=True))
        gamma = np.exp(joint - log_px)
        elbo = float(log_px.sum())
        elbo += float((fb * 0.5 * (np.log(inv_l) - inv_l - alpha * alpha + 1.0)).sum())
        pi = gamma.sum(axis=0)
        pi /= pi.sum()
        if prev_elbo != -np.inf and elbo - prev_elbo < epsilon:
            break
        prev_elbo = elbo

    labels = gamma.argmax(axis=1).astype(np.int32)
    return _compact_ids(labels)


def _smooth_binary_timeline(
    binary: np.ndarray,
    fill_short_gap_frames: int,
    min_region_frames: int,
) -> None:
    num_speakers = binary.shape[1]
    total_frames = binary.shape[0]
    for speaker_index in range(num_speakers):
        if fill_short_gap_frames > 0:
            _fill_short_false_gaps(binary, total_frames, speaker_index, fill_short_gap_frames)
        if min_region_frames > 1:
            _remove_short_true_regions(binary, total_frames, speaker_index, min_region_frames)


def _fill_short_false_gaps(binary: np.ndarray, total_frames: int, speaker_index: int, max_gap_frames: int) -> None:
    frame = 0
    while frame < total_frames:
        while frame < total_frames and not binary[frame, speaker_index]:
            frame += 1
        if frame >= total_frames:
            break
        while frame < total_frames and binary[frame, speaker_index]:
            frame += 1
        gap_start = frame
        while frame < total_frames and not binary[frame, speaker_index]:
            frame += 1
        if gap_start == 0 or frame >= total_frames:
            continue
        gap_length = frame - gap_start
        if gap_length <= max_gap_frames:
            binary[gap_start:frame, speaker_index] = True


def _remove_short_true_regions(binary: np.ndarray, total_frames: int, speaker_index: int, min_region_frames: int) -> None:
    frame = 0
    while frame < total_frames:
        while frame < total_frames and not binary[frame, speaker_index]:
            frame += 1
        if frame >= total_frames:
            break
        region_start = frame
        while frame < total_frames and binary[frame, speaker_index]:
            frame += 1
        region_length = frame - region_start
        if region_length <= min_region_frames:
            binary[region_start:frame, speaker_index] = False


def _merge_adjacent_segments(
    labeled: list[tuple[int, int, str]],
    frame_rate: int,
    merge_gap_frames: int,
) -> list[DiarizationSegment]:
    if not labeled:
        return []
    sorted_segments = sorted(labeled, key=lambda item: item[0])
    merged: list[DiarizationSegment] = []
    cur_start, cur_end, cur_speaker = sorted_segments[0]
    for start_frame, end_frame, speaker in sorted_segments[1:]:
        if speaker == cur_speaker and start_frame - cur_end <= merge_gap_frames:
            cur_end = max(cur_end, end_frame)
        else:
            merged.append(DiarizationSegment(cur_start / frame_rate, cur_end / frame_rate, cur_speaker))
            cur_start, cur_end, cur_speaker = start_frame, end_frame, speaker
    merged.append(DiarizationSegment(cur_start / frame_rate, cur_end / frame_rate, cur_speaker))
    return merged


def _create_reconstruction_debug_frames(total_frames: int) -> list[dict[str, object]] | None:
    window = _get_debug_reconstruction_window()
    if window is None:
        return None
    start_frame, end_frame = window
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))
    return [] if end_frame > start_frame else None


def _capture_reconstruction_debug_frame(
    debug_frames: list[dict[str, object]] | None,
    frame_index: int,
    average_count: float,
    selected_count: int,
    scores: np.ndarray,
    selected: np.ndarray,
) -> None:
    if debug_frames is None:
        return
    window = _get_debug_reconstruction_window()
    if window is None:
        return
    start_frame, end_frame = window
    if frame_index < start_frame or frame_index >= end_frame:
        return
    debug_frames.append(
        {
            "frame_index": int(frame_index),
            "time_seconds": frame_index / 50.0,
            "average_count": float(average_count),
            "selected_count": int(selected_count),
            "scores": [float(x) for x in scores.tolist()],
            "selected": [bool(x) for x in selected.tolist()],
        }
    )


def _write_reconstruction_debug_frames(debug_frames: list[dict[str, object]] | None) -> None:
    if debug_frames is None:
        return
    output_path = os.environ.get("DIARIZEN_ONNX_DEBUG_RECON_PATH")
    if not output_path:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(debug_frames, indent=2), encoding="utf-8")


def _get_debug_reconstruction_window() -> tuple[int, int] | None:
    raw = os.environ.get("DIARIZEN_ONNX_DEBUG_RECON_WINDOW")
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        return None
    try:
        start_seconds = float(parts[0])
        end_seconds = float(parts[1])
    except ValueError:
        return None
    start_frame = int(np.floor(start_seconds * 50))
    end_frame = int(np.ceil(end_seconds * 50))
    if end_frame <= start_frame:
        return None
    return start_frame, end_frame
