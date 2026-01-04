"""
Interfaz de usuario con Gradio
Pestañas: Text-to-Image, Image-to-Image, Extras
"""
import gradio as gr
import random
from datetime import datetime
from pathlib import Path
from PIL import Image
import config
from core import StableDiffusionManager, lora_manager


class ButterVisionUI:
    """Interfaz principal de ButterVision"""
    
    def __init__(self):
        self.sd_manager = StableDiffusionManager()
        self.lora_manager = lora_manager
        
    def txt2img_generate(
        self,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
        seed: int,
        num_images: int,
        scheduler: str,
        # LoRA parameters
        lora_1_name: str,
        lora_1_weight: float,
        lora_2_name: str,
        lora_2_weight: float,
    ):
        """Función de generación para txt2img"""
        try:
            # Cambiar scheduler si es necesario
            if scheduler != self.sd_manager.current_scheduler:
                self.sd_manager.change_scheduler(scheduler)
            
            # Cargar pipeline txt2img
            pipe = self.sd_manager.load_txt2img_pipeline()
            
            # Gestionar LoRAs
            self.lora_manager.unload_all_loras(pipe)
            
            if lora_1_name and lora_1_name != "None":
                self.lora_manager.load_lora(lora_1_name, lora_1_weight, pipe)
            
            if lora_2_name and lora_2_name != "None":
                self.lora_manager.load_lora(lora_2_name, lora_2_weight, pipe)
            
            # Generar seed aleatorio si es -1
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Generar imágenes
            images = self.sd_manager.generate_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                seed=seed,
                num_images=num_images,
            )
            
            # Guardar imágenes
            saved_paths = self._save_images(images, "txt2img", prompt, seed)
            
            info_text = (
                f"✅ Generación completada!\n"
                f"Seed: {seed}\n"
                f"Scheduler: {scheduler}\n"
                f"LoRAs activos: {list(self.lora_manager.get_loaded_loras().keys())}\n"
                f"Guardado en: {saved_paths[0].parent if saved_paths else 'N/A'}"
            )
            
            return images, info_text
            
        except Exception as e:
            error_msg = f"❌ Error en generación: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def img2img_generate(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        strength: float,
        seed: int,
        scheduler: str,
    ):
        """Función de generación para img2img"""
        try:
            if init_image is None:
                return None, "❌ Por favor carga una imagen inicial"
            
            # Cambiar scheduler
            if scheduler != self.sd_manager.current_scheduler:
                self.sd_manager.change_scheduler(scheduler)
            
            # Generar seed aleatorio si es -1
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            # Generar
            images = self.sd_manager.generate_img2img(
                init_image=init_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                cfg_scale=cfg_scale,
                strength=strength,
                seed=seed,
            )
            
            # Guardar
            saved_paths = self._save_images(images, "img2img", prompt, seed)
            
            info_text = (
                f"✅ Transformación completada!\n"
                f"Seed: {seed}\n"
                f"Strength: {strength}\n"
                f"Scheduler: {scheduler}"
            )
            
            return images, info_text
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def _save_images(self, images, mode: str, prompt: str, seed: int) -> list:
        """Guarda imágenes en el directorio de outputs"""
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, img in enumerate(images):
            filename = f"{mode}_{timestamp}_{seed}_{idx}.png"
            filepath = config.OUTPUTS_DIR / filename
            
            # Guardar con metadata
            metadata = {
                "prompt": prompt,
                "seed": str(seed),
                "mode": mode,
            }
            img.save(filepath, pnginfo=self._create_png_info(metadata))
            saved_paths.append(filepath)
        
        return saved_paths
    
    def _create_png_info(self, metadata: dict):
        """Crea metadata PNG para guardar con la imagen"""
        from PIL import PngImagePlugin
        info = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            info.add_text(key, str(value))
        return info
    
    def refresh_loras(self):
        """Refresca la lista de LoRAs disponibles"""
        self.lora_manager.scan_lora_directory()
        available = ["None"] + self.lora_manager.get_available_loras()
        return gr.Dropdown(choices=available), gr.Dropdown(choices=available)
    
    def create_interface(self):
        """Crea la interfaz completa de Gradio"""
        
        # Obtener listas de opciones
        schedulers = config.get_available_schedulers()
        loras = ["None"] + self.lora_manager.get_available_loras()
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="ButterVision - Stable Diffusion WebUI",
            css=".gradio-container {max-width: 1400px !important}"
        ) as interface:
            
            gr.Markdown(
                """
                # 🎨 ButterVision - Stable Diffusion WebUI
                ### WebUI ligero y personalizado para Stable Diffusion
                """
            )
            
            with gr.Tabs():
                # ==================== TAB: TEXT-TO-IMAGE ====================
                with gr.Tab("📝 Text-to-Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            txt2img_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe la imagen que quieres generar...",
                                lines=3,
                            )
                            txt2img_negative = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="Lo que NO quieres ver...",
                                lines=2,
                                value="worst quality, low quality, blurry, artifacts",
                            )
                            
                            with gr.Accordion("⚙️ Parámetros", open=True):
                                txt2img_steps = gr.Slider(
                                    label="Steps",
                                    minimum=1,
                                    maximum=100,
                                    value=30,
                                    step=1,
                                )
                                txt2img_cfg = gr.Slider(
                                    label="CFG Scale",
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.5,
                                    step=0.5,
                                )
                                
                                with gr.Row():
                                    txt2img_width = gr.Slider(
                                        label="Width",
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64,
                                    )
                                    txt2img_height = gr.Slider(
                                        label="Height",
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64,
                                    )
                                
                                txt2img_seed = gr.Number(
                                    label="Seed (-1 para random)",
                                    value=-1,
                                    precision=0,
                                )
                                txt2img_num_images = gr.Slider(
                                    label="Número de imágenes",
                                    minimum=1,
                                    maximum=4,
                                    value=1,
                                    step=1,
                                )
                                txt2img_scheduler = gr.Dropdown(
                                    label="Scheduler",
                                    choices=schedulers,
                                    value=config.model_config.default_scheduler,
                                )
                            
                            with gr.Accordion("🎭 LoRAs", open=False):
                                with gr.Row():
                                    txt2img_lora1 = gr.Dropdown(
                                        label="LoRA 1",
                                        choices=loras,
                                        value="None",
                                    )
                                    txt2img_lora1_weight = gr.Slider(
                                        label="Peso",
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=1.0,
                                        step=0.1,
                                    )
                                
                                with gr.Row():
                                    txt2img_lora2 = gr.Dropdown(
                                        label="LoRA 2",
                                        choices=loras,
                                        value="None",
                                    )
                                    txt2img_lora2_weight = gr.Slider(
                                        label="Peso",
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=1.0,
                                        step=0.1,
                                    )
                                
                                refresh_loras_btn = gr.Button("🔄 Refrescar LoRAs")
                            
                            txt2img_generate_btn = gr.Button(
                                "🚀 Generate",
                                variant="primary",
                                size="lg",
                            )
                        
                        with gr.Column(scale=1):
                            txt2img_output = gr.Gallery(
                                label="Imágenes generadas",
                                show_label=True,
                                columns=2,
                                height=600,
                            )
                            txt2img_info = gr.Textbox(
                                label="Info",
                                lines=4,
                                interactive=False,
                            )
                    
                    # Eventos txt2img
                    txt2img_generate_btn.click(
                        fn=self.txt2img_generate,
                        inputs=[
                            txt2img_prompt,
                            txt2img_negative,
                            txt2img_steps,
                            txt2img_cfg,
                            txt2img_width,
                            txt2img_height,
                            txt2img_seed,
                            txt2img_num_images,
                            txt2img_scheduler,
                            txt2img_lora1,
                            txt2img_lora1_weight,
                            txt2img_lora2,
                            txt2img_lora2_weight,
                        ],
                        outputs=[txt2img_output, txt2img_info],
                    )
                    
                    refresh_loras_btn.click(
                        fn=self.refresh_loras,
                        inputs=[],
                        outputs=[txt2img_lora1, txt2img_lora2],
                    )
                
                # ==================== TAB: IMAGE-TO-IMAGE ====================
                with gr.Tab("🖼️ Image-to-Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            img2img_input = gr.Image(
                                label="Imagen inicial",
                                type="pil",
                                height=300,
                            )
                            img2img_prompt = gr.Textbox(
                                label="Prompt",
                                lines=3,
                            )
                            img2img_negative = gr.Textbox(
                                label="Negative Prompt",
                                lines=2,
                                value="worst quality, low quality",
                            )
                            
                            with gr.Accordion("⚙️ Parámetros", open=True):
                                img2img_strength = gr.Slider(
                                    label="Strength (nivel de transformación)",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.75,
                                    step=0.05,
                                )
                                img2img_steps = gr.Slider(
                                    label="Steps",
                                    minimum=1,
                                    maximum=100,
                                    value=30,
                                    step=1,
                                )
                                img2img_cfg = gr.Slider(
                                    label="CFG Scale",
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.5,
                                    step=0.5,
                                )
                                img2img_seed = gr.Number(
                                    label="Seed (-1 para random)",
                                    value=-1,
                                    precision=0,
                                )
                                img2img_scheduler = gr.Dropdown(
                                    label="Scheduler",
                                    choices=schedulers,
                                    value=config.model_config.default_scheduler,
                                )
                            
                            img2img_generate_btn = gr.Button(
                                "🚀 Transform",
                                variant="primary",
                                size="lg",
                            )
                        
                        with gr.Column(scale=1):
                            img2img_output = gr.Gallery(
                                label="Resultado",
                                show_label=True,
                                columns=1,
                                height=600,
                            )
                            img2img_info = gr.Textbox(
                                label="Info",
                                lines=4,
                                interactive=False,
                            )
                    
                    # Eventos img2img
                    img2img_generate_btn.click(
                        fn=self.img2img_generate,
                        inputs=[
                            img2img_input,
                            img2img_prompt,
                            img2img_negative,
                            img2img_steps,
                            img2img_cfg,
                            img2img_strength,
                            img2img_seed,
                            img2img_scheduler,
                        ],
                        outputs=[img2img_output, img2img_info],
                    )
                
                # ==================== TAB: EXTRAS ====================
                with gr.Tab("⚡ Extras"):
                    gr.Markdown(
                        """
                        ### Herramientas adicionales
                        
                        **Funcionalidades planeadas:**
                        - Upscaling de imágenes (ESRGAN, RealESRGAN)
                        - Inpainting / Outpainting
                        - Batch processing
                        - Configuración de modelo
                        
                        *Esta pestaña se expandirá en futuras versiones*
                        """
                    )
                    
                    with gr.Accordion("🔧 Configuración del sistema", open=True):
                        gr.Markdown(f"**Modelo actual:** `{self.sd_manager.model_id}`")
                        gr.Markdown(f"**Dispositivo:** `{self.sd_manager.device}`")
                        gr.Markdown(f"**LoRAs disponibles:** `{len(loras)-1}`")
                        
                        with gr.Row():
                            clear_cache_btn = gr.Button("🧹 Limpiar VRAM")
                            unload_pipes_btn = gr.Button("♻️ Descargar Pipelines")
                        
                        system_info = gr.Textbox(
                            label="Estado del sistema",
                            lines=3,
                            interactive=False,
                        )
                        
                        def clear_vram():
                            import torch
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                allocated = torch.cuda.memory_allocated() / 1024**3
                                reserved = torch.cuda.memory_reserved() / 1024**3
                                return f"✅ VRAM limpiada\nAllocated: {allocated:.2f}GB\nReserved: {reserved:.2f}GB"
                            return "✅ Cache limpiado (CPU mode)"
                        
                        def unload_all():
                            self.sd_manager.unload_pipeline("all")
                            return "✅ Todos los pipelines descargados"
                        
                        clear_cache_btn.click(
                            fn=clear_vram,
                            inputs=[],
                            outputs=[system_info],
                        )
                        
                        unload_pipes_btn.click(
                            fn=unload_all,
                            inputs=[],
                            outputs=[system_info],
                        )
            
            gr.Markdown(
                """
                ---
                **ButterVision** - Stable Diffusion WebUI ligero y personalizado | 
                [GitHub](https://github.com/tuusuario/buttervision) | 
                Optimizado para GPUs con baja VRAM
                """
            )
        
        return interface


def create_ui():
    """Función helper para crear la interfaz"""
    ui = ButterVisionUI()
    return ui.create_interface()
