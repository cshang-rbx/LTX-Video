"""
Gradio demo for the LTX Video generator that keeps the pipeline loaded.

This UI reuses the caching helpers from `batch_infer.py` so checkpoints are
loaded only once per pipeline configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from batch_infer import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PIPELINE_CONFIG,
    get_pipeline_bundle,
    run_inference,
)
from ltx_video.inference import InferenceConfig, logger as inference_logger


OUTPUT_DIRECTORY = Path("gradio_cached_outputs")
PIPELINE_CHOICES = [
    "configs/ltxv-13b-0.9.8-distilled.yaml",
    "configs/ltxv-2b-0.9.8-distilled.yaml",
]


def _validate_conditioning_inputs(
    conditioning_image_path: Optional[str],
    conditioning_strength: float,
    conditioning_start_frame: int,
    num_frames: int,
) -> Tuple[
    Optional[list[str]], Optional[list[float]], Optional[list[int]], Optional[str]
]:
    if not conditioning_image_path:
        return None, None, None, None

    conditioning_image = Path(conditioning_image_path)
    if not conditioning_image.exists():
        return None, None, None, f"Conditioning image not found: {conditioning_image}"

    if conditioning_start_frame < 0:
        return None, None, None, "Conditioning start frame must be zero or positive."

    if conditioning_start_frame >= num_frames:
        return (
            None,
            None,
            None,
            "Conditioning start frame must be smaller than total frames.",
        )

    return (
        [str(conditioning_image)],
        [float(conditioning_strength)],
        [int(conditioning_start_frame)],
        None,
    )


def generate_video(
    prompt: str,
    conditioning_image_path: Optional[str],
    num_frames: int,
    height: int,
    width: int,
    seed: int,
    frame_rate: int,
    pipeline_config: str,
    conditioning_strength: float,
    conditioning_start_frame: int,
    image_cond_noise_scale: float,
    offload_to_cpu: bool,
) -> Tuple[Optional[str], str]:
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        return None, "Prompt is required."

    try:
        num_frames = int(num_frames)
        height = int(height)
        width = int(width)
        seed = int(seed)
        frame_rate = int(frame_rate)
        conditioning_start_frame = int(conditioning_start_frame)
        conditioning_strength = float(conditioning_strength)
        image_cond_noise_scale = float(image_cond_noise_scale)
        offload_to_cpu = bool(offload_to_cpu)
    except (TypeError, ValueError) as exc:
        return None, f"Invalid numeric input: {exc}"

    pipeline_path = Path(pipeline_config).expanduser()
    if not pipeline_path.exists():
        return None, f"Pipeline config not found: {pipeline_path}"

    (
        conditioning_paths,
        conditioning_strengths,
        conditioning_start_frames,
        conditioning_error,
    ) = _validate_conditioning_inputs(
        conditioning_image_path,
        conditioning_strength,
        conditioning_start_frame,
        num_frames,
    )
    if conditioning_error:
        return None, conditioning_error

    try:
        bundle = get_pipeline_bundle(str(pipeline_path))
    except Exception as exc:  # noqa: BLE001 - surface errors in UI
        inference_logger.error("Failed to prepare pipeline bundle", exc_info=True)
        return None, f"Failed to load pipeline: {exc}"

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    config = InferenceConfig(
        prompt=prompt_text,
        pipeline_config=str(pipeline_path),
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        output_path=str(OUTPUT_DIRECTORY),
        conditioning_media_paths=conditioning_paths,
        conditioning_strengths=conditioning_strengths,
        conditioning_start_frames=conditioning_start_frames,
        image_cond_noise_scale=image_cond_noise_scale,
        offload_to_cpu=offload_to_cpu,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    )

    try:
        outputs = run_inference(config, bundle)
    except Exception as exc:  # noqa: BLE001 - surface errors in UI
        inference_logger.error("Inference failed", exc_info=True)
        return None, f"Generation failed: {exc}"

    if not outputs:
        return None, "Pipeline returned no outputs."

    primary_output = outputs[0]

    try:
        prompt_record_path = primary_output.with_suffix(".txt")
        prompt_record_path.write_text(prompt_text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - surface errors in UI
        inference_logger.warning("Failed to store prompt text: %s", exc)

    return str(primary_output), f"Saved to {primary_output}"


with gr.Blocks() as demo:
    gr.Markdown("## LTX Video Generation Demo (Cached Pipeline)")
    prompt_input = gr.Textbox(
        label="Prompt",
        placeholder="Describe the video you want to generate...",
        lines=3,
    )
    conditioning_image_input = gr.Image(
        label="Conditioning Image (optional)",
        type="filepath",
        height=300,
    )

    with gr.Accordion("Advanced settings", open=False):
        num_frames_input = gr.Slider(
            minimum=9,
            maximum=513,
            step=8,
            value=300,
            label="Number of frames",
        )
        height_input = gr.Slider(
            minimum=256,
            maximum=1024,
            step=32,
            value=480,
            label="Height",
        )
        width_input = gr.Slider(
            minimum=256,
            maximum=1280,
            step=32,
            value=720,
            label="Width",
        )
        seed_input = gr.Number(value=1, label="Seed", precision=0)
        frame_rate_input = gr.Slider(
            minimum=4,
            maximum=60,
            step=1,
            value=24,
            label="Frame rate",
        )
        pipeline_config_input = gr.Dropdown(
            value=DEFAULT_PIPELINE_CONFIG,
            choices=PIPELINE_CHOICES,
            label="Pipeline config path",
            allow_custom_value=True,
        )
        conditioning_strength_input = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            value=1.0,
            label="Conditioning strength",
        )
        conditioning_start_frame_input = gr.Slider(
            minimum=0,
            maximum=512,
            step=1,
            value=0,
            label="Conditioning start frame",
        )
        image_cond_noise_scale_input = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.15,
            label="Image conditioning noise scale",
        )
        offload_checkbox = gr.Checkbox(
            value=False,
            label="Offload to CPU when GPU memory is low",
        )

    generate_button = gr.Button("Generate")
    output_video = gr.Video(label="Generated video", height=400)
    status_box = gr.Textbox(label="Status", interactive=False)

    generate_button.click(
        fn=generate_video,
        inputs=[
            prompt_input,
            conditioning_image_input,
            num_frames_input,
            height_input,
            width_input,
            seed_input,
            frame_rate_input,
            pipeline_config_input,
            conditioning_strength_input,
            conditioning_start_frame_input,
            image_cond_noise_scale_input,
            offload_checkbox,
        ],
        outputs=[output_video, status_box],
    )


if __name__ == "__main__":
    demo.queue().launch(share=True)
