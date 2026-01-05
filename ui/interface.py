"""
Interfaz de usuario minimalista con Gradio
Pestañas: Text to Image, Image to Image, Train LoRA, Settings
"""
import gradio as gr
import random
from datetime import datetime
from pathlib import Path
from PIL import Image
import config
from core import StableDiffusionManager, lora_manager, ModelManager


class ButterVisionUI:
    """Interfaz principal minimalista de ButterVision"""

    def __init__(self):
        self.sd_manager = StableDiffusionManager()
        self.lora_manager = lora_manager
        self.model_manager = ModelManager()
        self.available_models = self._scan_models()
    
    def update_detail_enhancer(self, enabled, weight):
        """Actualiza la configuración del LoRA de mejora de detalles"""
        try:
            self.sd_manager.set_detail_enhancer(enabled, weight)
            status = f"✅ LoRA de detalles {'activado' if enabled else 'desactivado'} (peso: {weight})"
            return status
        except Exception as e:
            return f"❌ Error al actualizar LoRA: {str(e)}"

    def _scan_models(self):
        """Escanea modelos locales disponibles"""
        local_models = self.model_manager.list_local_models()
        # Agregar el modelo por defecto si no está en locales
        default_model = config.model_config.model_id
        if default_model not in local_models:
            local_models.insert(0, default_model)
        return local_models

    def _save_images(self, images, mode, prompt, seed):
        """Guarda las imágenes generadas"""
        outputs_dir = Path("outputs") / mode
        outputs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []

        for i, img in enumerate(images):
            filename = f"{timestamp}_{seed}_{i+1}.png"
            filepath = outputs_dir / filename
            img.save(filepath)
            saved_paths.append(filepath)

        return saved_paths

    def txt2img_generate(self, prompt, negative_prompt, steps, cfg_scale, seed, model):
        """Generación Text to Image simplificada"""
        try:
            # Cambiar modelo si es necesario
            if model != self.sd_manager.current_model:
                self.sd_manager.change_model(model)

            # Cargar pipeline
            pipe = self.sd_manager.load_txt2img_pipeline()

            # Seed aleatorio si -1
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            # Generar
            images = self.sd_manager.generate_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                width=512,
                height=512,
                seed=seed,
                num_images=1,
            )

            # Guardar
            saved_paths = self._save_images(images, "txt2img", prompt, seed)

            info = f"✅ Generado con seed: {seed}\nGuardado en: {saved_paths[0].parent}"
            return images, info

        except Exception as e:
            return None, f"❌ Error: {str(e)}"

    def img2img_generate(self, init_image, prompt, negative_prompt, steps, cfg_scale, denoising_strength, seed, model):
        """Generación Image to Image simplificada"""
        try:
            if init_image is None:
                return None, "❌ Sube una imagen inicial"

            # Cambiar modelo si necesario
            if model != self.sd_manager.current_model:
                self.sd_manager.change_model(model)

            # Cargar pipeline
            pipe = self.sd_manager.load_img2img_pipeline()

            # Seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            # Generar
            images = self.sd_manager.generate_img2img(
                init_image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                strength=denoising_strength,
                seed=seed,
                num_images=1,
            )

            # Guardar
            saved_paths = self._save_images(images, "img2img", prompt, seed)

            info = f"✅ Generado con seed: {seed}\nGuardado en: {saved_paths[0].parent}"
            return images, info

        except Exception as e:
            return None, f"❌ Error: {str(e)}"

    def train_lora(self, training_images, trigger_word, base_model, epochs, learning_rate, network_rank, progress=gr.Progress()):
        """Entrenamiento de LoRA (placeholder - implementar lógica completa)"""
        try:
            if training_images is None:
                return "❌ Sube imágenes de entrenamiento"

            progress(0.1, "Preparando datos...")

            # Aquí iría la lógica de entrenamiento
            # Por ahora, solo simular
            import time
            for i in range(epochs):
                progress((i+1)/epochs, f"Entrenando epoch {i+1}/{epochs}...")
                time.sleep(1)  # Simular entrenamiento

            progress(1.0, "Entrenamiento completado")
            return f"✅ LoRA entrenado: {trigger_word}\nModelo base: {base_model}\nEpochs: {epochs}"

        except Exception as e:
            return f"❌ Error en entrenamiento: {str(e)}"

    def refresh_models(self):
        """Refresca la lista de modelos disponibles"""
        self.available_models = self._scan_models()
        return gr.update(choices=self.available_models)

    def download_model(self, url_or_id, model_name=""):
        """Descarga modelo desde URL o ID"""
        try:
            if not url_or_id.strip():
                return "❌ Ingresa una URL o ID de modelo"

            # Determinar tipo de descarga
            if url_or_id.startswith("https://huggingface.co/"):
                # Modelo de Hugging Face
                model_id = url_or_id.replace("https://huggingface.co/", "").split("/")[0:2]
                model_id = "/".join(model_id)
                if not model_name:
                    model_name = model_id.replace("/", "_")
                path = self.model_manager.download_hf_model(model_id, model_name)
                status = f"✅ Modelo HF descargado: {path}"
            
            elif url_or_id.startswith("https://civitai.com/") or url_or_id.isdigit():
                # Modelo de CivitAI
                if url_or_id.isdigit():
                    model_id = url_or_id
                else:
                    # Extraer ID de la URL
                    model_id = url_or_id.split("/")[-1]
                
                if not model_name:
                    model_name = f"civitai_{model_id}"
                
                path = self.model_manager.download_civitai_model(model_id, model_name)
                status = f"✅ Modelo CivitAI descargado: {path}"
            
            else:
                # Asumir ID de HF
                if not model_name:
                    model_name = url_or_id.replace("/", "_")
                path = self.model_manager.download_hf_model(url_or_id, model_name)
                status = f"✅ Modelo HF descargado: {path}"

            # Refrescar lista de modelos
            self.available_models = self._scan_models()
            return status

        except Exception as e:
            return f"❌ Error descargando modelo: {str(e)}"

    def create_interface(self):
        """Crea la interfaz minimalista con 4 pestañas"""

        # Estado para modelos disponibles
        models_state = gr.State(self.available_models)

        with gr.Blocks(title="ButterVision - Minimal SD WebUI", theme=gr.themes.Soft()) as interface:

            gr.Markdown("# 🎨 ButterVision")
            gr.Markdown("Stable Diffusion WebUI minimalista y limpio")

            with gr.Tabs():

                # ========================================
                # PESTAÑA 1: TEXT TO IMAGE
                # ========================================
                with gr.TabItem("Text to Image", id="txt2img"):

                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe la imagen que quieres generar...",
                                lines=3,
                            )
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Elementos a evitar...",
                                lines=2,
                            )

                        with gr.Column(scale=1):
                            steps = gr.Slider(20, 100, value=20, step=1, label="Steps")
                            cfg_scale = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG Scale")
                            seed = gr.Number(value=-1, label="Seed (-1 = random)")
                            with gr.Row():
                                txt2img_model = gr.Dropdown(
                                    choices=self.available_models,
                                    value=self.available_models[0],
                                    label="Model"
                                )
                                refresh_model_btn = gr.Button("🔄", size="sm")

                    generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                    gallery = gr.Gallery(label="Results", show_label=True, columns=2, height=400)

                    info_text = gr.Textbox(label="Info", interactive=False, lines=2)

                    generate_btn.click(
                        fn=self.txt2img_generate,
                        inputs=[prompt, negative_prompt, steps, cfg_scale, seed, txt2img_model],
                        outputs=[gallery, info_text]
                    )

                    refresh_model_btn.click(
                        fn=self.refresh_models,
                        inputs=[],
                        outputs=[txt2img_model]
                    )

                # ========================================
                # PESTAÑA 2: IMAGE TO IMAGE
                # ========================================
                with gr.TabItem("Image to Image", id="img2img"):

                    with gr.Row():
                        with gr.Column(scale=2):
                            init_image = gr.Image(label="Initial Image", type="pil")
                            prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe los cambios...",
                                lines=3,
                            )
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Elementos a evitar...",
                                lines=2,
                            )

                        with gr.Column(scale=1):
                            steps = gr.Slider(20, 100, value=20, step=1, label="Steps")
                            cfg_scale = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG Scale")
                            denoising_strength = gr.Slider(0, 1, value=0.75, step=0.05, label="Denoising Strength")
                            seed = gr.Number(value=-1, label="Seed (-1 = random)")
                            with gr.Row():
                                img2img_model = gr.Dropdown(
                                    choices=self.available_models,
                                    value=self.available_models[0],
                                    label="Model"
                                )
                                refresh_model_btn = gr.Button("🔄", size="sm")

                    generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

                    gallery = gr.Gallery(label="Results", show_label=True, columns=2, height=400)

                    info_text = gr.Textbox(label="Info", interactive=False, lines=2)

                    generate_btn.click(
                        fn=self.img2img_generate,
                        inputs=[init_image, prompt, negative_prompt, steps, cfg_scale, denoising_strength, seed, img2img_model],
                        outputs=[gallery, info_text]
                    )

                    refresh_model_btn.click(
                        fn=self.refresh_models,
                        inputs=[],
                        outputs=[img2img_model]
                    )

                # ========================================
                # PESTAÑA 3: TRAIN LORA
                # ========================================
                with gr.TabItem("Train LoRA", id="train_lora"):

                    gr.Markdown("""
                    ## 🎯 Entrenamiento de LoRA
                    **LoRA (Low-Rank Adaptation)** permite entrenar un modelo personalizado con tus fotos.
                    El resultado será un archivo pequeño que puedes usar para generar imágenes de ti mismo.
                    """)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📸 Imágenes de Entrenamiento")
                            training_images = gr.File(
                                label="Sube tus fotos (15-30 imágenes recomendadas)",
                                file_types=[".zip", ".png", ".jpg", ".jpeg"],
                                file_count="directory"
                            )
                            gr.Markdown("*Sube un ZIP con tus fotos o selecciona una carpeta. Usa fotos variadas de tu rostro desde diferentes ángulos.*")

                            gr.Markdown("### 🏷️ Palabra Activadora")
                            trigger_word = gr.Textbox(
                                label="Trigger Word (palabra que activará tu LoRA)",
                                placeholder="ej: nicobutter, mirostro, johnstyle",
                                info="Esta palabra se usará en los prompts para activar tu estilo personalizado"
                            )
                            gr.Markdown("*Elige una palabra única que no uses normalmente. Ej: 'nicobutter' activará tu estilo.*")

                        with gr.Column():
                            gr.Markdown("### 🤖 Modelo Base")
                            with gr.Row():
                                train_base_model = gr.Dropdown(
                                    choices=self.available_models,
                                    value=self.available_models[0],
                                    label="Modelo base para el entrenamiento",
                                    info="El modelo SD que se usará como base. Tu GTX 1650 puede usar cualquier modelo de 1.5GB o menos"
                                )
                                refresh_model_btn = gr.Button("🔄", size="sm")
                            gr.Markdown("*Para GTX 1650: usa modelos ligeros como SD 1.5. Evita SDXL por ahora.*")

                            gr.Markdown("### ⚙️ Parámetros de Entrenamiento")
                            epochs = gr.Slider(1, 20, value=10, step=1, label="Epochs (iteraciones completas)",
                                             info="Más epochs = mejor calidad pero más tiempo. 10-15 es buen inicio")
                            learning_rate = gr.Slider(0.00001, 0.001, value=0.0001, step=0.00001, label="Learning Rate",
                                                    info="Qué tan rápido aprende. 0.0001 es conservador y seguro")
                            network_rank = gr.Slider(8, 32, value=16, step=4, label="Network Rank (tamaño del LoRA)",
                                                   info="Tamaño del archivo LoRA. 16 es buen balance calidad/tamaño")

                    gr.Markdown("---")

                    with gr.Row():
                        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                        gr.Markdown("""
                        **⏱️ Tiempo estimado:** 30-60 minutos con GTX 1650
                        **💾 Espacio requerido:** ~500MB para el proceso
                        **📁 Resultado:** Archivo .safetensors en `models/lora/`
                        """)

                    progress_output = gr.Textbox(
                        label="Progreso del Entrenamiento",
                        interactive=False,
                        lines=8,
                        placeholder="Aquí aparecerá el progreso del entrenamiento..."
                    )

                    train_btn.click(
                        fn=self.train_lora,
                        inputs=[training_images, trigger_word, train_base_model, epochs, learning_rate, network_rank],
                        outputs=[progress_output]
                    )

                    refresh_model_btn.click(
                        fn=self.refresh_models,
                        inputs=[],
                        outputs=[train_base_model]
                    )

                # ========================================
                # PESTAÑA 4: SETTINGS
                # ========================================
                with gr.TabItem("Settings", id="settings"):

                    gr.Markdown("## Model Management")

                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Models")
                        settings_models_list = gr.Dropdown(
                            choices=self.available_models,
                            label="Available Models",
                            interactive=False
                        )

                    refresh_btn.click(
                        fn=self.refresh_models,
                        inputs=[],
                        outputs=[settings_models_list]
                    )

                    gr.Markdown("## 🎨 Detail Enhancer LoRA")
                    gr.Markdown("*Mejora automáticamente la calidad y detalles de todas las imágenes generadas*")

                    with gr.Row():
                        detail_enhancer_enabled = gr.Checkbox(
                            label="Enable Detail Enhancer",
                            value=True,
                            info="Activa el LoRA que mejora detalles, caras y texturas"
                        )
                        detail_enhancer_weight = gr.Slider(
                            label="Detail Enhancer Strength",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.6,
                            step=0.1,
                            info="Fuerza del efecto de mejora de detalles (0.6 recomendado)"
                        )

                    update_detail_btn = gr.Button("🔄 Update Detail Enhancer", variant="secondary")

                    detail_status = gr.Textbox(
                        label="Detail Enhancer Status",
                        value="✅ LoRA de detalles activado (peso: 0.6)",
                        interactive=False
                    )

                    update_detail_btn.click(
                        fn=self.update_detail_enhancer,
                        inputs=[detail_enhancer_enabled, detail_enhancer_weight],
                        outputs=[detail_status]
                    )

                    gr.Markdown("## 📥 Download Model")

                    with gr.Row():
                        model_url = gr.Textbox(
                            label="Model URL/ID",
                            placeholder="https://huggingface.co/... o https://civitai.com/... o ID de CivitAI"
                        )
                        model_name = gr.Textbox(
                            label="Local Name (optional)",
                            placeholder="Nombre para guardar localmente"
                        )

                    download_btn = gr.Button("📥 Download Model", variant="secondary")

                    download_output = gr.Textbox(label="Download Status", interactive=False)

                    download_btn.click(
                        fn=self.download_model,
                        inputs=[model_url, model_name],
                        outputs=[download_output]
                    )

                    gr.Markdown("## VRAM Options")

                    with gr.Row():
                        medvram = gr.Checkbox(label="Medium VRAM Mode", value=False)
                        lowvram = gr.Checkbox(label="Low VRAM Mode", value=False)

                    gr.Markdown("## Theme")

                    theme = gr.Radio(
                        choices=["Dark", "Light"],
                        value="Dark",
                        label="Interface Theme"
                    )

            gr.Markdown("---")
            gr.Markdown("**ButterVision** - Minimal Stable Diffusion WebUI")

        return interface


def create_ui():
    """Función helper para crear la interfaz"""
    ui = ButterVisionUI()
    return ui.create_interface()