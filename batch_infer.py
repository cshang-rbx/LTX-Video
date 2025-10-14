"""Batch generation script for LTX Video.

Loads the model stack once and runs inference for a list of prompt/image pairs
so that checkpoints are not reloaded between runs.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from ltx_video.inference import (
    InferenceConfig,
    calculate_padding,
    create_latent_upsampler,
    create_ltx_video_pipeline,
    get_total_gpu_memory,
    get_unique_filename,
    load_media_file,
    load_pipeline_config,
    prepare_conditioning,
    seed_everething,
    logger as inference_logger,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline


DEFAULT_PIPELINE_CONFIG = "configs/ltxv-13b-0.9.8-distilled.yaml"
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)
DEFAULT_TASKS: List[Dict[str, object]] = []


@dataclass
class PipelineBundle:
    pipeline: object
    raw_config: Dict
    device: str
    model_path: str
    prompt_enhancer_image_caption_model_name: Optional[str]
    prompt_enhancer_llm_model_name: Optional[str]
    prompt_enhancers_loaded: bool = False


_PIPELINE_CACHE: Dict[Path, PipelineBundle] = {}


def _canonical_config_path(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().resolve()


def _resolve_model_path(model_name_or_path: str) -> str:
    candidate = Path(model_name_or_path)
    if candidate.is_file():
        return str(candidate)
    return hf_hub_download(
        repo_id="Lightricks/LTX-Video",
        filename=model_name_or_path,
        repo_type="model",
    )


def _get_base_pipeline(pipeline: object) -> object:
    if isinstance(pipeline, LTXMultiScalePipeline):
        return pipeline.video_pipeline
    return pipeline


def _ensure_prompt_enhancers(bundle: PipelineBundle, enhance_prompt: bool) -> None:
    if not enhance_prompt or bundle.prompt_enhancers_loaded:
        return

    caption_name = bundle.prompt_enhancer_image_caption_model_name
    llm_name = bundle.prompt_enhancer_llm_model_name
    if not caption_name or not llm_name:
        # No additional models configured; nothing to do.
        return

    base_pipeline = _get_base_pipeline(bundle.pipeline)

    caption_model = AutoModelForCausalLM.from_pretrained(
        caption_name, trust_remote_code=True
    )
    caption_processor = AutoProcessor.from_pretrained(
        caption_name, trust_remote_code=True
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.bfloat16
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)

    base_pipeline.prompt_enhancer_image_caption_model = caption_model
    base_pipeline.prompt_enhancer_image_caption_processor = caption_processor
    base_pipeline.prompt_enhancer_llm_model = llm_model
    base_pipeline.prompt_enhancer_llm_tokenizer = llm_tokenizer
    bundle.prompt_enhancers_loaded = True


def get_pipeline_bundle(config_path: str) -> PipelineBundle:
    key = _canonical_config_path(config_path)
    inference_logger.info("Preparing pipeline bundle for %s", key)

    if key in _PIPELINE_CACHE:
        inference_logger.info("Using cached pipeline bundle for %s", key)
        return _PIPELINE_CACHE[key]

    config_dict = load_pipeline_config(config_path)
    inference_logger.info("Loaded pipeline configuration from %s", config_path)
    raw_config = copy.deepcopy(config_dict)

    model_path = _resolve_model_path(config_dict["checkpoint_path"])
    inference_logger.info("Resolved checkpoint to %s", model_path)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    inference_logger.info("Initializing pipeline on device %s", device)
    pipeline = create_ltx_video_pipeline(
        ckpt_path=model_path,
        precision=config_dict["precision"],
        text_encoder_model_name_or_path=config_dict["text_encoder_model_name_or_path"],
        sampler=config_dict.get("sampler"),
        device=device,
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path=config_dict.get(
            "prompt_enhancer_image_caption_model_name_or_path"
        ),
        prompt_enhancer_llm_model_name_or_path=config_dict.get(
            "prompt_enhancer_llm_model_name_or_path"
        ),
    )

    if config_dict.get("pipeline_type") == "multi-scale":
        upsampler_path = config_dict.get("spatial_upscaler_model_path")
        if upsampler_path is None:
            raise ValueError(
                "Multi-scale pipeline requires 'spatial_upscaler_model_path' in the config."
            )
        inference_logger.info("Loading latent upsampler from %s", upsampler_path)
        upsampler_model = _resolve_model_path(upsampler_path)
        latent_upsampler = create_latent_upsampler(upsampler_model, device)
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler)
        inference_logger.info("Wrapping base pipeline with multi-scale upsampler")

    bundle = PipelineBundle(
        pipeline=pipeline,
        raw_config=raw_config,
        device=device,
        model_path=model_path,
        prompt_enhancer_image_caption_model_name=config_dict.get(
            "prompt_enhancer_image_caption_model_name_or_path"
        ),
        prompt_enhancer_llm_model_name=config_dict.get(
            "prompt_enhancer_llm_model_name_or_path"
        ),
    )

    _PIPELINE_CACHE[key] = bundle
    inference_logger.info("Cached pipeline bundle for %s", key)
    return bundle


def _prepare_runtime_kwargs(raw_config: Dict) -> Tuple[Dict, str, float, str]:
    config_copy = copy.deepcopy(raw_config)
    stg_mode = config_copy.pop("stg_mode", "attention_values")
    prompt_threshold = config_copy.get("prompt_enhancement_words_threshold", 0)
    precision = config_copy.get("precision", "fp32")

    for field in (
        "checkpoint_path",
        "spatial_upscaler_model_path",
        "text_encoder_model_name_or_path",
        "precision",
        "sampler",
        "prompt_enhancement_words_threshold",
        "prompt_enhancer_image_caption_model_name_or_path",
        "prompt_enhancer_llm_model_name_or_path",
        "pipeline_type",
    ):
        config_copy.pop(field, None)

    return config_copy, stg_mode, prompt_threshold, precision


def _select_skip_layer_strategy(stg_mode: str):
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

    mode = stg_mode.lower()
    if mode in {"stg_av", "attention_values"}:
        return SkipLayerStrategy.AttentionValues
    if mode in {"stg_as", "attention_skip"}:
        return SkipLayerStrategy.AttentionSkip
    if mode in {"stg_r", "residual"}:
        return SkipLayerStrategy.Residual
    if mode in {"stg_t", "transformer_block"}:
        return SkipLayerStrategy.TransformerBlock
    raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")


def run_inference(config: InferenceConfig, bundle: PipelineBundle) -> List[Path]:
    runtime_kwargs, stg_mode, prompt_threshold, precision = _prepare_runtime_kwargs(
        bundle.raw_config
    )
    inference_logger.info(
        "Running inference: prompt='%s', seed=%d, %dx%d, frames=%d",
        config.prompt,
        config.seed,
        config.height,
        config.width,
        config.num_frames,
    )

    prompt_word_count = len(config.prompt.split())
    enhance_prompt = prompt_threshold > 0 and prompt_word_count < prompt_threshold
    _ensure_prompt_enhancers(bundle, enhance_prompt)

    conditioning_media_paths = config.conditioning_media_paths
    conditioning_strengths = config.conditioning_strengths
    conditioning_start_frames = config.conditioning_start_frames

    if conditioning_media_paths:
        if not conditioning_strengths:
            conditioning_strengths = [1.0] * len(conditioning_media_paths)
        if not conditioning_start_frames:
            raise ValueError(
                "If `conditioning_media_paths` is provided, `conditioning_start_frames` must also be provided"
            )
        if len(conditioning_media_paths) != len(conditioning_strengths) or len(
            conditioning_media_paths
        ) != len(conditioning_start_frames):
            raise ValueError(
                "`conditioning_media_paths`, `conditioning_strengths`, and `conditioning_start_frames` must have the same length"
            )
        if any(s < 0 or s > 1 for s in conditioning_strengths):
            raise ValueError("All conditioning strengths must be between 0 and 1")
        if any(f < 0 or f >= config.num_frames for f in conditioning_start_frames):
            raise ValueError(
                f"All conditioning start frames must be between 0 and {config.num_frames - 1}"
            )

    seed_everething(config.seed)

    if config.offload_to_cpu and not torch.cuda.is_available():
        inference_logger.warning(
            "offload_to_cpu is True, but CUDA is unavailable; running on CPU without offloading."
        )
        offload_to_cpu = False
    else:
        offload_to_cpu = config.offload_to_cpu and get_total_gpu_memory() < 30

    inference_logger.info(
        "Offload to CPU resolved to %s (requested=%s)",
        offload_to_cpu,
        config.offload_to_cpu,
    )
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_logger.info("Writing outputs to %s", output_dir)

    height_padded = ((config.height - 1) // 32 + 1) * 32
    width_padded = ((config.width - 1) // 32 + 1) * 32
    num_frames_padded = ((config.num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(config.height, config.width, height_padded, width_padded)
    inference_logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

    media_item = None
    if config.input_media_path:
        inference_logger.info("Loading input media from %s", config.input_media_path)
        media_item = load_media_file(
            media_path=config.input_media_path,
            height=config.height,
            width=config.width,
            max_frames=num_frames_padded,
            padding=padding,
        )

    conditioning_items = (
        prepare_conditioning(
            conditioning_media_paths=conditioning_media_paths,
            conditioning_strengths=conditioning_strengths,
            conditioning_start_frames=conditioning_start_frames,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            padding=padding,
            pipeline=bundle.pipeline,
        )
        if conditioning_media_paths
        else None
    )

    skip_layer_strategy = _select_skip_layer_strategy(stg_mode)
    sample = {
        "prompt": config.prompt,
        "prompt_attention_mask": None,
        "negative_prompt": config.negative_prompt,
        "negative_prompt_attention_mask": None,
    }

    base_pipeline = _get_base_pipeline(bundle.pipeline)
    execution_device = getattr(base_pipeline, "_execution_device", None)
    generator_device = execution_device if execution_device is not None else bundle.device
    if isinstance(generator_device, torch.device):
        generator_device = generator_device if generator_device.index is not None else generator_device.type
    if isinstance(generator_device, str) and generator_device.startswith("cuda") and not torch.cuda.is_available():
        generator_device = "cpu"
    if offload_to_cpu:
        generator_device = "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(config.seed)
    inference_logger.info("Invoking pipeline with generator on %s", generator_device)
    pipeline_output = bundle.pipeline(
        **runtime_kwargs,
        skip_layer_strategy=skip_layer_strategy,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=config.frame_rate,
        **sample,
        media_items=media_item,
        conditioning_items=conditioning_items,
        is_video=True,
        vae_per_channel_normalize=True,
        image_cond_noise_scale=config.image_cond_noise_scale,
        mixed_precision=(precision == "mixed_precision"),
        offload_to_cpu=offload_to_cpu,
        device=bundle.device,
        enhance_prompt=enhance_prompt,
    )

    images = pipeline_output.images
    inference_logger.info("Pipeline returned tensor with shape %s", tuple(images.shape))

    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, : config.num_frames, pad_top:pad_bottom, pad_left:pad_right]

    saved_paths: List[Path] = []
    for i in range(images.shape[0]):
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        fps = config.frame_rate
        height, width = video_np.shape[1:3]

        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            output_filename = get_unique_filename(
                f"video_output_{i}",
                ".mp4",
                prompt=config.prompt,
                seed=config.seed,
                resolution=(height, width, config.num_frames),
                dir=output_dir,
            )
            with imageio.get_writer(output_filename, fps=fps) as video_writer:
                for frame in video_np:
                    video_writer.append_data(frame)

        saved_paths.append(output_filename)
        inference_logger.warning(f"Output saved to {output_filename}")

    return saved_paths


def _load_tasks(path: Optional[str]) -> List[Dict[str, object]]:
    if path is None:
        inference_logger.info("No tasks file provided; using DEFAULT_TASKS with %d items", len(DEFAULT_TASKS))
        return list(DEFAULT_TASKS)

    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {task_path}")

    inference_logger.info("Loading tasks from %s", task_path)
    with open(task_path, "r", encoding="utf-8") as handle:
        if task_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)

    if not isinstance(data, Iterable):
        raise ValueError("Tasks file must contain a list of task objects")

    tasks: List[Dict[str, object]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Task #{idx} is not an object: {item}")
        if "prompt" not in item:
            raise ValueError(f"Task #{idx} missing required 'prompt' field")
        tasks.append(item)
    inference_logger.info("Loaded %d tasks from %s", len(tasks), task_path)
    return tasks


def _resolve_conditioning(task: Dict[str, object], defaults: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[List[float]], Optional[List[int]]]:
    image = task.get("image") or task.get("conditioning_image")
    if image is None:
        return None, None, None

    image_path = Path(str(image)).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Conditioning image not found: {image_path}")

    strength = float(task.get("conditioning_strength", defaults.conditioning_strength))
    start_frame = int(task.get("conditioning_start_frame", defaults.conditioning_start_frame))
    return [str(image_path)], [strength], [start_frame]


def build_config(task: Dict[str, object], defaults: argparse.Namespace) -> InferenceConfig:
    conditioning_paths, conditioning_strengths, conditioning_start_frames = _resolve_conditioning(
        task, defaults
    )

    output_root = Path(defaults.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    task_dir = output_root / task.get("output_folder", f"task_{defaults.task_index:03d}")

    config = InferenceConfig(
        prompt=str(task["prompt"]),
        pipeline_config=defaults.config,
        seed=int(task.get("seed", defaults.seed)),
        height=int(task.get("height", defaults.height)),
        width=int(task.get("width", defaults.width)),
        num_frames=int(task.get("num_frames", defaults.num_frames)),
        frame_rate=int(task.get("frame_rate", defaults.frame_rate)),
        output_path=str(task_dir),
        conditioning_media_paths=conditioning_paths,
        conditioning_strengths=conditioning_strengths,
        conditioning_start_frames=conditioning_start_frames,
        image_cond_noise_scale=float(
            task.get("image_cond_noise_scale", defaults.image_cond_noise_scale)
        ),
        offload_to_cpu=bool(task.get("offload_to_cpu", defaults.offload_to_cpu)),
        negative_prompt=str(task.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
    )

    if "input_media_path" in task:
        config.input_media_path = str(task["input_media_path"])

    task_index = getattr(defaults, "task_index", None)
    if task_index is not None:
        inference_logger.info(
            "Built configuration for task %s with output directory %s",
            task_index,
            task_dir,
        )
    else:
        inference_logger.info(
            "Built configuration for prompt '%s' with output directory %s",
            config.prompt,
            task_dir,
        )

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch LTX Video generation without Gradio UI")
    parser.add_argument(
        "--config",
        default=DEFAULT_PIPELINE_CONFIG,
        help="Pipeline config YAML (defaults to configs/ltxv-13b-0.9.8-distilled.yaml)",
    )
    parser.add_argument(
        "--tasks-file",
        default=None,
        help="JSON or YAML file containing a list of tasks with prompts and optional settings.",
    )
    parser.add_argument(
        "--output-dir",
        default="batch_outputs",
        help="Root directory where task outputs will be stored.",
    )
    parser.add_argument("--height", type=int, default=480, help="Default output height")
    parser.add_argument("--width", type=int, default=720, help="Default output width")
    parser.add_argument("--num-frames", type=int, default=300, help="Default number of frames")
    parser.add_argument("--frame-rate", type=int, default=24, help="Default frame rate")
    parser.add_argument("--seed", type=int, default=1, help="Default random seed")
    parser.add_argument(
        "--conditioning-strength",
        type=float,
        default=1.0,
        help="Default conditioning strength when an image is provided.",
    )
    parser.add_argument(
        "--conditioning-start-frame",
        type=int,
        default=0,
        help="Default frame index to start conditioning.",
    )
    parser.add_argument(
        "--image-cond-noise-scale",
        type=float,
        default=0.15,
        help="Default image conditioning noise scale.",
    )
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Attempt to offload to CPU when GPU memory is low.",
    )

    args = parser.parse_args()
    inference_logger.info(
        "Starting batch job with config=%s, tasks_file=%s, output_dir=%s",
        args.config,
        args.tasks_file,
        args.output_dir,
    )
    tasks = _load_tasks(args.tasks_file)
    inference_logger.info("Task loader returned %d tasks", len(tasks))
    if not tasks:
        raise SystemExit("No tasks to process. Provide a tasks file or edit DEFAULT_TASKS.")

    inference_logger.info("Fetching pipeline bundle for %s", args.config)
    bundle = get_pipeline_bundle(args.config)

    inference_logger.info("Beginning execution of %d tasks", len(tasks))
    for index, task in enumerate(tasks, start=1):
        inference_logger.warning("Starting task %s/%s: %s", index, len(tasks), task.get("prompt"))
        args.task_index = index
        config = build_config(task, args)
        outputs = run_inference(config, bundle)
        for path in outputs:
            print(f"Task {index}: saved {path}")


if __name__ == "__main__":
    # Safeguard: allow execution via `python gradio_demo2.py`
    main()
