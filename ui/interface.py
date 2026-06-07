"""
Interfaz MVP de ButterVision.
Solo expone Text-to-Image para estabilizar el flujo base.
"""
import random
from datetime import datetime
from pathlib import Path

import gradio as gr

import config
from core.advanced_pipeline import ButterVisionPipeline


class ButterVisionUI:
    """Interfaz principal enfocada únicamente en Text-to-Image."""

    def __init__(self):
        self.sd_manager = ButterVisionPipeline(
            model_id=config.model_config.model_id,
            enable_optimizations=True,
            enable_lcm=False,
        )

    def _save_images(self, images, seed):
        """Guarda imágenes generadas en outputs/txt2img."""
        outputs_dir = config.OUTPUTS_DIR / "txt2img"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []

        for index, image in enumerate(images, start=1):
            filepath = outputs_dir / f"{timestamp}_{seed}_{index}.png"
            image.save(filepath)
            saved_paths.append(filepath)

        return saved_paths

    def txt2img_generate(
        self,
        prompt,
        negative_prompt,
        steps,
        cfg_scale,
        width,
        height,
        seed,
    ):
        """Genera una imagen desde texto."""
        try:
            prompt = (prompt or "").strip()
            negative_prompt = (negative_prompt or "").strip()

            if not prompt:
                return None, "Ingresa un prompt para generar una imagen."

            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = int(seed)

            images = self.sd_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(cfg_scale),
                width=int(width),
                height=int(height),
                seed=seed,
                num_images=1,
            )

            saved_paths = self._save_images(images, seed)
            info = (
                f"Seed: {seed}\n"
                f"Size: {int(width)}x{int(height)}\n"
                f"Steps: {int(steps)} | CFG: {float(cfg_scale):.1f}\n"
                f"Saved: {saved_paths[0]}"
            )
            return images[0], info

        except Exception as error:
            return None, f"Error: {error}"

    def create_interface(self):
        """Crea la interfaz Text-to-Image."""
        with gr.Blocks(title="ButterVision - Text to Image") as interface:
            gr.Markdown("# ButterVision")

            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe la imagen que quieres generar...",
                        lines=4,
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Elementos a evitar...",
                        lines=3,
                    )

                    generate_btn = gr.Button("Generate", variant="primary", size="lg")

                with gr.Column(scale=1):
                    steps = gr.Slider(
                        minimum=1,
                        maximum=60,
                        value=config.model_config.default_steps,
                        step=1,
                        label="Steps",
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=config.model_config.default_cfg_scale,
                        step=0.5,
                        label="CFG Scale",
                    )
                    width = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=config.model_config.default_width,
                        step=64,
                        label="Width",
                    )
                    height = gr.Slider(
                        minimum=256,
                        maximum=768,
                        value=config.model_config.default_height,
                        step=64,
                        label="Height",
                    )
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

            with gr.Row():
                image_output = gr.Image(type="pil", label="Generated Image", height=512)
                info_text = gr.Textbox(label="Info", interactive=False, lines=5)

            generate_btn.click(
                fn=self.txt2img_generate,
                inputs=[
                    prompt,
                    negative_prompt,
                    steps,
                    cfg_scale,
                    width,
                    height,
                    seed,
                ],
                outputs=[image_output, info_text],
            )

        return interface


def create_ui():
    """Crea la interfaz de ButterVision."""
    ui = ButterVisionUI()
    return ui.create_interface()
