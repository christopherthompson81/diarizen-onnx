#!/usr/bin/env python3
"""
Export pyannote/wespeaker-voxceleb-resnet34-LM to ONNX.

This is the weighted WeSpeaker model used by the current DiariZen ONNX
pipeline in this repository.
- Input:  Kaldi Fbank features (batch, time_frames, 80) — 80 mel bins, 10 ms shift
- Output: raw 256-dim speaker embedding

Two export variants are supported:
- default: Fbank only
- weighted: Fbank + frame weights, matching pyannote's native
  `model_(waveforms, weights=masks)` path more closely

The model's internal Fbank parameters (must match C# WeSpeakerEmbedder):
  num_mel_bins  = 80
  frame_length  = 25 ms  -> 400 samples at 16 kHz
  frame_shift   = 10 ms  -> 160 samples at 16 kHz
  dither        = 0.0
  window_type   = "hamming"
  use_energy    = False
  waveform_scale = 32768 (multiply before kaldi.fbank)

Usage:
    python scripts/export_pyannote_wespeaker_onnx.py \
        --output-dir ./models [--test]
"""

import argparse
import importlib.util
import os
import pkgutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from torch.jit import TracerWarning


def resolve_diarizen_root(explicit_root: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root)

    env_root = os.environ.get("DIARIZEN_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "DiariZen",
            Path("/home/chris/Programming/DiariZen"),
        ]
    )

    for root in candidates:
        if root.exists() and (root / "diarizen").exists():
            return root.resolve()

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "DiariZen repository not found. Pass --diarizen-root or set DIARIZEN_ROOT.\n"
        f"Searched:\n{searched}"
    )


def setup_paths(diarizen_root: Path) -> None:
    sys.path.insert(0, str(diarizen_root))
    print(f"Using DiariZen from: {diarizen_root}")

    local_pyannote_root = diarizen_root / "pyannote-audio" / "pyannote"
    local_pyannote_init = local_pyannote_root / "__init__.py"
    if not local_pyannote_init.exists():
        raise FileNotFoundError(f"Local pyannote package not found at {local_pyannote_init}")

    spec = importlib.util.spec_from_file_location("pyannote", local_pyannote_init)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pyannote namespace from {local_pyannote_init}")
    pyannote = importlib.util.module_from_spec(spec)
    sys.modules["pyannote"] = pyannote
    spec.loader.exec_module(pyannote)
    pyannote.__path__ = pkgutil.extend_path([str(local_pyannote_root)], pyannote.__name__)


class FbankEmbeddingWrapper(nn.Module):
    """Wraps the pyannote WeSpeakerResNet34 to take pre-computed Fbank.

    Raw output is required so that the LDA/VBx transform (which expects un-normalised embeddings)
    can be applied correctly in C#. The C# pipeline applies L2-normalisation after the LDA transform.
    """

    def __init__(self, model, with_weights: bool = False):
        super().__init__()
        self.resnet = model.resnet
        self.with_weights = with_weights

    def forward(
        self, fbank: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        fbank : (batch, time_frames, 80)  Kaldi Fbank features
        weights : (batch, time_frames) optional
            Frame weights used by pyannote's native embedding path.

        Returns
        -------
        embedding : (batch, 256)  raw (un-normalised)
        """
        if not self.with_weights:
            _, embed = self.resnet(fbank)
            return embed

        x = fbank.permute(0, 2, 1)
        x = x.unsqueeze(1)
        out = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
        out = self.resnet.layer1(out)
        out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)

        if weights is not None:
            weights = F.interpolate(
                weights.unsqueeze(1), size=out.shape[-1], mode="nearest"
            ).squeeze(1)

        stats = self.resnet.pool(out, weights=weights)
        embed = self.resnet.seg_1(stats)
        return embed  # No normalisation — let C# apply LDA + L2-norm


def load_model():
    from pyannote.audio import Model
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        filename="pytorch_model.bin",
    )
    print(f"Loading from {ckpt_path}")
    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_torch_load
    try:
        model = Model.from_pretrained(ckpt_path)
    finally:
        torch.load = original_torch_load
    model.eval()

    # Verify hyperparameters
    hp = model.hparams
    print(f"  num_mel_bins={hp.num_mel_bins}, frame_length={hp.frame_length}ms, "
          f"frame_shift={hp.frame_shift}ms, window={hp.window_type}")
    assert hp.num_mel_bins == 80, f"Unexpected num_mel_bins={hp.num_mel_bins}"

    # Check output dim
    with torch.no_grad():
        dummy_fbank = torch.zeros(1, 249, 80)  # ~2.5s
        _, embed = model.resnet(dummy_fbank)
    print(f"  Embedding dim: {embed.shape[-1]}")
    assert embed.shape[-1] == 256, f"Unexpected embed dim {embed.shape[-1]}"

    return model


def set_symbolic_batch_dimension(model_path: Path) -> None:
    """Fix exported ONNX shape metadata so ORT does not warn on batched output."""
    model_proto = onnx.load(str(model_path))
    changed = False

    for value_info in model_proto.graph.output:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = tensor_type.shape.dim
        if len(dims) < 2:
            continue
        batch_dim = dims[0]
        if batch_dim.HasField("dim_value") or not batch_dim.HasField("dim_param"):
            batch_dim.ClearField("dim_value")
            batch_dim.dim_param = "batch"
            changed = True

    if changed:
        onnx.save(model_proto, str(model_path))


def export_model(wrapper: FbankEmbeddingWrapper, output_path: Path):
    print(f"\nExporting to {output_path} ...")
    wrapper.eval()
    dummy = torch.zeros(1, 249, 80)
    dummy_weights = torch.ones(1, 249)

    with torch.no_grad():
        out = wrapper(dummy, dummy_weights if wrapper.with_weights else None)
    print(f"  PyTorch: input {tuple(dummy.shape)} -> output {tuple(out.shape)}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Converting a tensor to a Python boolean might cause the trace to be incorrect.*",
            category=TracerWarning,
        )

        if wrapper.with_weights:
            torch.onnx.export(
                wrapper,
                (dummy, dummy_weights),
                str(output_path),
                input_names=["fbank", "weights"],
                output_names=["embedding"],
                dynamic_axes={
                    "fbank":     {0: "batch", 1: "frames"},
                    "weights":   {0: "batch", 1: "frames"},
                    "embedding": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )
        else:
            torch.onnx.export(
                wrapper,
                dummy,
                str(output_path),
                input_names=["fbank"],
                output_names=["embedding"],
                dynamic_axes={
                    "fbank":     {0: "batch", 1: "frames"},
                    "embedding": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )

    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    set_symbolic_batch_dimension(output_path)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  OK: {size_mb:.1f} MB")
    print("  Input:  (batch, time_frames, 80) — Kaldi Fbank, 10ms shift, 25ms window")
    print("  Output: (batch, 256)             — raw embedding (un-normalised; LDA transform applied in C#)")


def test_model(onnx_path: Path, wrapper: FbankEmbeddingWrapper):
    import onnxruntime as ort

    print("\nVerifying ONNX output matches PyTorch (batch=1) ...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    np.random.seed(42)
    test_input = np.random.randn(1, 249, 80).astype(np.float32)
    test_weights = np.random.rand(1, 249).astype(np.float32)

    if wrapper.with_weights:
        ort_out = sess.run(None, {"fbank": test_input, "weights": test_weights})[0]
    else:
        ort_out = sess.run(None, {"fbank": test_input})[0]
    with torch.no_grad():
        pt_out = wrapper(
            torch.from_numpy(test_input),
            torch.from_numpy(test_weights) if wrapper.with_weights else None,
        ).numpy()

    np.testing.assert_allclose(ort_out, pt_out, rtol=1e-4, atol=1e-4)
    print(f"  ✓ Max diff: {np.abs(ort_out - pt_out).max():.2e}  shape: {ort_out.shape}")


def main():
    parser = argparse.ArgumentParser(description="Export pyannote WeSpeaker to ONNX (80-bin Fbank input)")
    parser.add_argument("--output-dir", type=Path, default=Path("./models"))
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--diarizen-root",
        type=Path,
        default=None,
        help="Path to a local DiariZen checkout (defaults to DIARIZEN_ROOT or common local paths)",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Export a variant that also accepts frame weights.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diarizen_root = resolve_diarizen_root(args.diarizen_root)
    setup_paths(diarizen_root)

    model = load_model()
    wrapper = FbankEmbeddingWrapper(model, with_weights=args.weighted)
    output_path = (
        args.output_dir / "wespeaker_pyannote_weighted.onnx"
        if args.weighted
        else args.output_dir / "wespeaker_pyannote.onnx"
    )

    export_model(wrapper, output_path)

    if args.test:
        test_model(output_path, wrapper)

    print(f"\nDone: {output_path}")
    print("\nFbank parameters for C# WeSpeakerEmbedder:")
    print("  NumMelBins  = 80")
    print("  FrameLength = 400 samples (25ms at 16kHz)")
    print("  FrameShift  = 160 samples (10ms at 16kHz)")
    print("  Scale       = 32768 (multiply waveform before computing Fbank)")
    print("  Window      = Hamming")
    print("  UseEnergy   = false")
    print("  Dither      = 0.0")


if __name__ == "__main__":
    main()
