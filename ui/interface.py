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
from core.model_manager import ModelManager


UI_CSS = """
html {
    scroll-behavior: smooth;
}
body,
.gradio-container {
    background:
        radial-gradient(circle at 15% 8%, rgba(0, 229, 255, 0.18), transparent 28%),
        radial-gradient(circle at 84% 18%, rgba(255, 43, 214, 0.16), transparent 28%),
        linear-gradient(135deg, #070812 0%, #0b1020 48%, #12081b 100%) !important;
    color: #e5f7ff !important;
}
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 18px 18px 40px !important;
}
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.55), transparent 75%);
}
.topbar {
    position: sticky;
    top: 12px;
    z-index: 20;
    align-items: center;
    border: 1px solid rgba(0, 229, 255, 0.24);
    border-radius: 8px;
    padding: 11px 12px;
    background: rgba(8, 13, 28, 0.82);
    box-shadow:
        0 18px 45px rgba(0, 0, 0, 0.38),
        inset 0 0 0 1px rgba(255,255,255,0.04),
        0 0 26px rgba(0, 229, 255, 0.08);
    backdrop-filter: blur(18px);
    margin-bottom: 18px;
    transition: padding 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
}
.topbar.bv-shrink {
    padding: 6px 10px;
    border-color: rgba(255, 43, 214, 0.34);
    box-shadow:
        0 12px 32px rgba(0, 0, 0, 0.46),
        inset 0 0 0 1px rgba(255,255,255,0.04),
        0 0 22px rgba(255, 43, 214, 0.10);
}
.brand {
    font-size: 20px;
    font-weight: 800;
    color: #f8fbff;
    line-height: 1;
    padding-top: 8px;
    letter-spacing: 0;
    text-shadow: 0 0 16px rgba(0, 229, 255, 0.42);
}
.brand-accent {
    color: #00e5ff;
}
.topbar .form {
    border: 0;
    background: transparent;
    padding: 0;
}
.topbar label {
    font-size: 11px;
    color: #93a4b8;
}
.topbar .model-status {
    min-height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 700;
    white-space: nowrap;
    padding: 0 12px;
    border: 1px solid;
}
.model-status-ok {
    color: #96ffd1;
    background: rgba(0, 255, 170, 0.11);
    border-color: rgba(0, 255, 170, 0.38);
    box-shadow: 0 0 18px rgba(0, 255, 170, 0.12);
}
.model-status-bad {
    color: #ffb4c8;
    background: rgba(255, 43, 84, 0.12);
    border-color: rgba(255, 43, 84, 0.42);
    box-shadow: 0 0 18px rgba(255, 43, 84, 0.12);
}
.model-status-warn {
    color: #ffe7a3;
    background: rgba(255, 198, 41, 0.11);
    border-color: rgba(255, 198, 41, 0.35);
}
.bv-panel,
.bv-output {
    border: 1px solid rgba(0, 229, 255, 0.16);
    border-radius: 8px;
    background: linear-gradient(180deg, rgba(13, 20, 38, 0.86), rgba(8, 12, 24, 0.92));
    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.38),
        inset 0 0 0 1px rgba(255,255,255,0.035);
    padding: 16px;
}
.bv-panel h2,
.bv-output h2 {
    margin: 0 0 12px;
    color: #f8fbff;
    font-size: 16px;
    font-weight: 800;
}
.bv-panel label,
.bv-output label {
    color: #b9c9dc !important;
}
.bv-panel textarea,
.bv-panel input,
.bv-panel .wrap,
.bv-output textarea {
    background: rgba(4, 9, 20, 0.68) !important;
    color: #e8f8ff !important;
    border-color: rgba(0, 229, 255, 0.20) !important;
}
.bv-panel textarea:focus,
.bv-panel input:focus {
    border-color: rgba(0, 229, 255, 0.62) !important;
    box-shadow: 0 0 0 1px rgba(0, 229, 255, 0.30), 0 0 20px rgba(0, 229, 255, 0.12) !important;
}
.bv-panel .gr-button-primary {
    border: 1px solid rgba(0, 229, 255, 0.68) !important;
    background: linear-gradient(90deg, #00e5ff, #ff2bd6) !important;
    color: #050712 !important;
    font-weight: 900 !important;
    box-shadow: 0 0 28px rgba(0, 229, 255, 0.20);
}
.bv-panel .gr-button-primary:hover {
    filter: brightness(1.08);
    box-shadow: 0 0 36px rgba(255, 43, 214, 0.26);
}
.bv-params {
    border-color: rgba(255, 43, 214, 0.20);
}
.bv-output {
    margin-top: 18px;
}
.bv-output .image-container,
.bv-output img {
    border-radius: 8px !important;
}
@media (max-width: 780px) {
    .gradio-container {
        padding: 10px 10px 28px !important;
    }
    .topbar {
        top: 6px;
    }
    .brand {
        font-size: 18px;
    }
    .topbar .model-status {
        justify-content: flex-start;
    }
}
"""

UI_JS = """
() => {
  const updateTopbar = () => {
    const topbar = document.querySelector('.topbar');
    if (!topbar) return;
    topbar.classList.toggle('bv-shrink', window.scrollY > 36);
  };
  window.addEventListener('scroll', updateTopbar, { passive: true });
  updateTopbar();
}
"""


class ButterVisionUI:
    """Interfaz principal enfocada únicamente en Text-to-Image."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.model_choices = self._get_model_choices()
        active_model = self.model_manager.resolve_model_path(config.model_config.model_id) or config.model_config.model_id
        self.sd_manager = ButterVisionPipeline(
            model_id=active_model,
            enable_optimizations=True,
            enable_lcm=False,
        )

    def _get_model_choices(self):
        """Retorna modelos locales detectados y asegura que el activo esté presente."""
        choices = [model["path"] for model in self.model_manager.list_local_model_infos()]
        active_model = config.model_config.model_id
        resolved_active = self.model_manager.resolve_model_path(active_model) or active_model

        if resolved_active not in choices:
            choices.insert(0, resolved_active)

        return choices

    def _format_model_label(self, model_path):
        """Muestra un nombre breve para el selector."""
        path = Path(model_path)
        return path.stem if path.suffix else path.name

    def _model_status_html(self, model_id):
        """Genera indicador visual de compatibilidad modelo/VRAM."""
        info = self.model_manager.get_model_info(model_id)
        if info is None:
            return (
                "<div class='model-status model-status-warn'>"
                "Modelo no encontrado"
                "</div>"
            )

        if info["fits_gpu"] is True:
            return "<div class='model-status model-status-ok'>Compatible con la GPU</div>"
        elif info["fits_gpu"] is False:
            return "<div class='model-status model-status-bad'>No compatible con la GPU</div>"

        return "<div class='model-status model-status-warn'>Compatibilidad no determinada</div>"

    def refresh_models(self):
        """Actualiza lista de modelos locales y el indicador."""
        self.model_choices = self._get_model_choices()
        current_model = self.sd_manager.model_id
        if current_model not in self.model_choices:
            current_model = self.model_choices[0] if self.model_choices else current_model
        return (
            gr.update(choices=self.model_choices, value=current_model),
            self._model_status_html(current_model),
        )

    def select_model(self, model_id):
        """Cambia el modelo activo para la próxima generación."""
        if not model_id:
            return self._model_status_html(self.sd_manager.model_id)

        if model_id != self.sd_manager.model_id:
            self.sd_manager.change_model(model_id)
            config.model_config.model_id = model_id

        return self._model_status_html(model_id)

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

            active_model = self.sd_manager.model_id
            with gr.Row(elem_classes=["topbar"]):
                with gr.Column(scale=1, min_width=150):
                    gr.HTML("<div class='brand'>Butter<span class='brand-accent'>Vision</span></div>")
                with gr.Column(scale=4, min_width=320):
                    model_selector = gr.Dropdown(
                        choices=self.model_choices,
                        value=active_model,
                        label="Modelo actual",
                    )
                with gr.Column(scale=0, min_width=56):
                    refresh_models_btn = gr.Button("↻", size="sm")
                with gr.Column(scale=2, min_width=220):
                    model_status = gr.HTML(value=self._model_status_html(active_model))

            with gr.Row():
                with gr.Column(scale=2, elem_classes=["bv-panel", "bv-prompts"]):
                    gr.HTML("<h2>Prompt</h2>")
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

                with gr.Column(scale=1, elem_classes=["bv-panel", "bv-params"]):
                    gr.HTML("<h2>Parámetros</h2>")
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

            with gr.Column(elem_classes=["bv-output"]):
                gr.HTML("<h2>Generated Image</h2>")
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

            model_selector.change(
                fn=self.select_model,
                inputs=[model_selector],
                outputs=[model_status],
            )

            refresh_models_btn.click(
                fn=self.refresh_models,
                inputs=[],
                outputs=[model_selector, model_status],
            )

        return interface


def create_ui():
    """Crea la interfaz de ButterVision."""
    ui = ButterVisionUI()
    return ui.create_interface()


def get_ui_css():
    """CSS de la interfaz."""
    return UI_CSS


def get_ui_js():
    """JS de la interfaz."""
    return UI_JS
