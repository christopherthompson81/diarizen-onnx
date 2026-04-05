from __future__ import annotations

import argparse
import sys

from .diarizer import DiariZenDiarizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal DiariZen ONNX reference diarizer")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--model-dir", required=True, help="Directory containing ONNX and PLDA files")
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda"], help="ONNX Runtime provider")
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--ahc-threshold", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    args = parser.parse_args()

    diarizer = DiariZenDiarizer(args.model_dir, provider=args.provider)
    segments = diarizer.diarize_file(
        args.audio,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        threshold=args.threshold,
        ahc_threshold=args.ahc_threshold,
    )

    if args.json:
        sys.stdout.write(diarizer.segments_to_json(segments))
        sys.stdout.write("\n")
        return 0

    for segment in segments:
        sys.stdout.write(f"{segment.start:8.2f} {segment.end:8.2f} {segment.speaker}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
