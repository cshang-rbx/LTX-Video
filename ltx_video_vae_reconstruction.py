#!/usr/bin/env python3
"""
Minimal example for running the LTX-Video causal VAE on a video clip.

This script reads a video file, optionally resizes it, makes it compatible with
the VAE's temporal and spatial strides, encodes the clip into latents, decodes
them back to pixel space, and writes a reconstruction that matches the original
frame count, resolution, and frame rate.
"""

import argparse
import math
import os
import time
from typing import Optional, Tuple
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from safetensors import safe_open

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)

try:
    import imageio.v3 as iio  # type: ignore
except ImportError:  # pragma: no cover - imageio.v3 is preferred but optional
    import imageio as iio  # type: ignore


def _load_video(path: str) -> Tuple[torch.Tensor, Optional[float]]:
    """Load a video file and return a (C, T, H, W) tensor plus an optional FPS."""
    frames = iio.imread(path, index=None)
    if frames.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor, got shape {frames.shape}.")
    if frames.shape[-1] != 3:
        raise ValueError("Only RGB videos are supported.")
    tensor = torch.from_numpy(frames).float() / 127.5 - 1.0  # map to [-1, 1]
    fps = None
    try:
        meta = iio.immeta(path)
        fps = meta.get("fps")
    except Exception:
        pass
    return tensor.permute(3, 0, 1, 2), fps


def _save_video(video: torch.Tensor, path: str, fps: float) -> None:
    """Save a (C, T, H, W) tensor to disk."""
    video = video.clamp(-1.0, 1.0)
    frames = (
        ((video + 1.0) * 127.5)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    kwargs = {"fps": fps} if fps else {}
    try:
        iio.imwrite(path, frames, **kwargs)
    except TypeError:
        iio.mimwrite(path, frames, fps=fps if fps else 24)  # type: ignore[attr-defined]


def _resize_video(video: torch.Tensor, size: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Resize a (C, T, H, W) video to (width, height) if requested."""
    if size is None:
        return video
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Resize dimensions must be positive.")
    if video.shape[-1] == width and video.shape[-2] == height:
        return video
    video = F.interpolate(
        video.permute(1, 0, 2, 3),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return video.permute(1, 0, 2, 3)


def _match_frame_count(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Trim or pad the temporal dimension to match `target_frames`."""
    current = video.shape[1]
    if current == target_frames:
        return video
    if current > target_frames:
        return video[:, :target_frames]
    if current == 0:
        raise ValueError("Input video has no frames after preprocessing.")
    pad = target_frames - current
    last = video[:, -1:].repeat(1, pad, 1, 1)
    return torch.cat([video, last], dim=1)


def _make_stride_compatible(
    video: torch.Tensor,
    t_stride: int,
    s_stride: int,
    mode: str,
) -> torch.Tensor:
    """Adjust temporal and spatial sizes to satisfy stride constraints."""
    c, t, h, w = video.shape

    def _trim(value: int, stride: int) -> int:
        trimmed = value // stride * stride
        return max(trimmed, stride)

    def _pad(value: int, stride: int) -> int:
        padded = math.ceil(value / stride) * stride
        return max(padded, stride)

    need_t_fix = t_stride > 1 and t % t_stride != 0
    need_h_fix = s_stride > 1 and h % s_stride != 0
    need_w_fix = s_stride > 1 and w % s_stride != 0

    if mode == "error":
        if need_t_fix:
            raise ValueError(
                f"Video length {t} is not compatible with the temporal stride {t_stride}. "
                f"Trim or pad the clip so that frames are divisible by {t_stride}."
            )
        if need_h_fix or need_w_fix:
            raise ValueError(
                f"Spatial size {(h, w)} must be divisible by the spatial stride {s_stride}."
            )
        return video

    if mode not in {"trim", "pad"}:
        raise ValueError(f"Unsupported stride_compat option '{mode}'.")

    new_t = t
    new_h = h
    new_w = w
    if need_t_fix:
        new_t = (_trim if mode == "trim" else _pad)(t, t_stride)
    if need_h_fix:
        new_h = (_trim if mode == "trim" else _pad)(h, s_stride)
    if need_w_fix:
        new_w = (_trim if mode == "trim" else _pad)(w, s_stride)

    if mode == "trim":
        video = video[:, :new_t, :new_h, :new_w]
    else:
        pad_t = new_t - t
        pad_h = new_h - h
        pad_w = new_w - w
        if pad_t > 0:
            last = video[:, -1:].repeat(1, pad_t, 1, 1)
            video = torch.cat([video, last], dim=1)
        if pad_h > 0 or pad_w > 0:
            pad_args = (0, pad_w, 0, pad_h)
            video = F.pad(video, pad_args, mode="replicate")
    return video


def _resolve_vae_checkpoint_path(path: str) -> str:
    """Return a checkpoint location that `CausalVideoAutoencoder` can understand."""
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file() or candidate.suffix != ".safetensors":
        return path

    metadata_config = None
    try:
        with safe_open(str(candidate), framework="pt") as handle:
            metadata = handle.metadata() or {}
            metadata_config = metadata.get("config")
    except Exception:
        metadata_config = None

    if metadata_config:
        return path

    parent = candidate.parent
    if (parent / "config.json").is_file():
        return str(candidate.parent.parent)

    if parent.name == "vae":
        root = parent.parent
        if root != parent and (root / "vae" / "config.json").is_file():
            return str(root)

    return path


def _extract_default_fps(ckpt_path: str) -> Optional[float]:
    """Read checkpoint metadata to find a default FPS if available."""
    if not os.path.isfile(ckpt_path):
        return None
    try:
        with safe_open(ckpt_path, framework="pt") as f:
            metadata = f.metadata()
    except Exception:
        return None
    if metadata is None:
        return None
    config_str = metadata.get("config")
    if not config_str:
        return None
    try:
        config = json.loads(config_str)
    except Exception:
        return None

    def _pull_fps(cfg: dict) -> Optional[float]:
        for key in ("sample_fps", "fps", "video_fps", "frame_rate"):
            if key in cfg:
                try:
                    return float(cfg[key])
                except (TypeError, ValueError):
                    continue
        video_cfg = cfg.get("video")
        if isinstance(video_cfg, dict):
            return _pull_fps(video_cfg)
        return None

    return _pull_fps(config)


def _select_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_reconstruction(
    ckpt_path: str,
    video_path: str,
    output_path: str,
    device: torch.device,
    fps: Optional[float] = None,
    stride_compat: str = "pad",
    resize: Optional[Tuple[int, int]] = None,
    use_bfloat16: bool = False,
) -> None:
    timings = {}

    load_start = time.perf_counter()
    video_cpu, input_fps = _load_video(video_path)
    orig_frames = video_cpu.shape[1]
    orig_size = (video_cpu.shape[-1], video_cpu.shape[-2])
    timings["input_loading"] = time.perf_counter() - load_start

    model_start = time.perf_counter()
    resolved_ckpt = _resolve_vae_checkpoint_path(ckpt_path)
    print(f"{resolved_ckpt=} {ckpt_path=}")
    vae = CausalVideoAutoencoder.from_pretrained(resolved_ckpt)
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    timings["model_loading"] = time.perf_counter() - model_start

    temporal_stride = getattr(vae, "temporal_downscale_factor", 1)
    spatial_stride = getattr(vae, "spatial_downscale_factor", 1)

    preprocess_start = time.perf_counter()
    video_proc = _resize_video(video_cpu, resize)
    video_proc = _make_stride_compatible(
        video_proc, temporal_stride, spatial_stride, stride_compat
    )
    video_batch = video_proc.unsqueeze(0).to(device=device, dtype=dtype)
    timings["preprocessing"] = time.perf_counter() - preprocess_start

    infer_start = time.perf_counter()
    with torch.inference_mode():
        encoding_start = time.perf_counter()
        posterior = vae.encode(video_batch)
        latents = posterior.latent_dist.mode()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["model_encoding"] = time.perf_counter() - encoding_start

        decoding_start = time.perf_counter()
        recon = vae.decode(latents, target_shape=video_batch.shape).sample
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["model_decoding"] = time.perf_counter() - decoding_start
    timings["model_inference"] = time.perf_counter() - infer_start

    recon = recon.squeeze(0).to(torch.float32).cpu()
    recon = _match_frame_count(recon, orig_frames)
    recon = _resize_video(recon, orig_size)

    reference_fps = float(
        fps
        or input_fps
        or _extract_default_fps(ckpt_path)
        or 24.0
    )

    output_start = time.perf_counter()
    _save_video(recon, output_path, reference_fps)
    timings["output_saving"] = time.perf_counter() - output_start

    print(f"Original input shape: {(3, orig_frames, orig_size[1], orig_size[0])}")
    print(f"Preprocessed input shape: {tuple(video_proc.shape)}")
    print(f"Latent shape: {tuple(latents.shape)}")
    print(f"Output video shape: {tuple(recon.shape)}")
    print(f"Output FPS: {reference_fps}")
    print(f"Temporal stride: {temporal_stride}, Spatial stride: {spatial_stride}")
    print(f"Reconstruction saved to: {output_path}")
    print(
        "Timings (s): "
        f"input={timings['input_loading']:.2f}, "
        f"model_load={timings['model_loading']:.2f}, "
        f"preprocess={timings['preprocessing']:.2f}, "
        f"inference={timings['model_inference']:.2f}, "
        f"encode={timings['model_encoding']:.2f}, "
        f"decode={timings['model_decoding']:.2f}, "
        f"save={timings['output_saving']:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LTX-Video VAE for encode/decode reconstruction."
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the LTX-Video checkpoint (.safetensors) or directory with VAE weights.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Input video to encode and decode.",
    )
    parser.add_argument(
        "--output",
        default="ltx_vae_reconstruction.mp4",
        help="Where to write the reconstructed video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override for the output video.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU even if CUDA/MPS are available.",
    )
    parser.add_argument(
        "--stride-compat",
        choices=["error", "trim", "pad"],
        default="pad",
        help="How to handle sizes that do not match the VAE stride.",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize input to WIDTHxHEIGHT before encoding (e.g. 1280x704).",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Run the VAE in bfloat16 for reduced memory usage.",
    )
    return parser.parse_args()


def _parse_resize_arg(resize: Optional[str]) -> Optional[Tuple[int, int]]:
    if resize is None:
        return None
    value = resize.strip().lower()
    if value in {"", "none"}:
        return None
    if "x" not in value and "*" not in value:
        raise ValueError("Resize format must be WIDTHxHEIGHT, e.g. 1280x704.")
    width_str, height_str = value.split("x", 1) if "x" in value else value.split("*", 1)
    return (int(width_str), int(height_str))


def main() -> None:
    args = parse_args()
    resize = _parse_resize_arg(args.resize)
    device = _select_device(args.cpu)
    run_reconstruction(
        ckpt_path=args.ckpt,
        video_path=args.video,
        output_path=args.output,
        device=device,
        fps=args.fps,
        stride_compat=args.stride_compat,
        resize=resize,
        use_bfloat16=args.bfloat16,
    )


if __name__ == "__main__":
    main()
