"""
Gradio demo for the LTX Video generator.

The interface wraps the existing `ltx_video.inference.infer` entry point while
managing prompt/image inputs and saving artifacts with a shared stem.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from ltx_video.inference import InferenceConfig, infer


DEFAULT_PIPELINE_CONFIG = "configs/ltxv-2b-0.9.8-distilled.yaml"
OUTPUT_DIRECTORY = Path("gradio_outputs")
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)


def _slugify(text: str) -> str:
    cleaned = []
    for char in text.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "-", "_"}:
            cleaned.append("-")
    # Collapse multiple separators and trim
    slug = "-".join(filter(None, "".join(cleaned).split("-")))
    return slug[:80]


def _resolve_unique_stem(base_dir: Path, base_stem: str) -> str:
    candidate = base_stem or f"ltx-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if not (base_dir / f"{candidate}.mp4").exists():
        return candidate

    counter = 1
    while True:
        new_candidate = f"{candidate}_{counter}"
        if not (base_dir / f"{new_candidate}.mp4").exists():
            return new_candidate
        counter += 1


def _collect_generated_media(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.glob("*"),
        key=lambda file_path: file_path.stat().st_mtime,
    )
    for candidate in candidates:
        if candidate.suffix.lower() in {".mp4", ".png"}:
            return candidate
    raise FileNotFoundError("No media file generated during inference.")


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

    num_frames = int(num_frames)
    height = int(height)
    width = int(width)
    seed = int(seed)
    frame_rate = int(frame_rate)
    conditioning_start_frame = int(conditioning_start_frame)
    conditioning_strength = float(conditioning_strength)
    image_cond_noise_scale = float(image_cond_noise_scale)
    offload_to_cpu = bool(offload_to_cpu)

    conditioning_image = (
        Path(conditioning_image_path) if conditioning_image_path else None
    )
    if conditioning_image and not conditioning_image.exists():
        return None, f"Conditioning image not found: {conditioning_image}"

    if conditioning_image and conditioning_start_frame >= num_frames:
        return None, "Conditioning start frame must be smaller than total frames."

    pipeline_path = Path(pipeline_config)
    if not pipeline_path.exists():
        return (
            None,
            f"Pipeline config not found: {pipeline_config}",
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIRECTORY / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        config = InferenceConfig(
            prompt=prompt_text,
            pipeline_config=pipeline_config,
            seed=int(seed),
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=int(frame_rate),
            output_path=str(run_dir),
            conditioning_media_paths=[str(conditioning_image)]
            if conditioning_image
            else None,
            conditioning_strengths=[float(conditioning_strength)]
            if conditioning_image
            else None,
            conditioning_start_frames=[int(conditioning_start_frame)]
            if conditioning_image
            else None,
            image_cond_noise_scale=float(image_cond_noise_scale),
            offload_to_cpu=bool(offload_to_cpu),
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        )

        infer(config=config)

        generated_media = _collect_generated_media(run_dir)

        base_stem = _slugify(prompt_text)
        if not base_stem and conditioning_image:
            base_stem = conditioning_image.stem.lower()
        stem = _resolve_unique_stem(OUTPUT_DIRECTORY, base_stem)

        final_media_path = OUTPUT_DIRECTORY / f"{stem}{generated_media.suffix.lower()}"
        generated_media.replace(final_media_path)

        prompt_text_path = OUTPUT_DIRECTORY / f"{stem}.txt"
        prompt_text_path.write_text(prompt_text, encoding="utf-8")

        if conditioning_image:
            target_conditioning_path = OUTPUT_DIRECTORY / (
                f"{stem}{conditioning_image.suffix.lower()}"
            )
            if conditioning_image != target_conditioning_path:
                shutil.copyfile(conditioning_image, target_conditioning_path)

        # Cleanup the run directory if it is empty.
        if not any(run_dir.iterdir()):
            run_dir.rmdir()

        return str(final_media_path), f"Saved to {final_media_path}"
    except Exception as exc:  # noqa: BLE001 - surface errors in UI
        return None, f"Generation failed: {exc}"


with gr.Blocks() as demo:
    gr.Markdown("## LTX Video Generation Demo")
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
            choices=["configs/ltxv-13b-0.9.8-distilled.yaml", "configs/ltxv-2b-0.9.8-distilled.yaml"],
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
