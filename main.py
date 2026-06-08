#!/usr/bin/env python3
"""
ButterVision - Main Entry Point
Stable Diffusion WebUI ligero y personalizado

Uso:
    python main.py [opciones]

Ejemplos:
    python main.py
    python main.py --share
    python main.py --port 7861 --model "stabilityai/stable-diffusion-2-1"
    python main.py --lowvram
"""
import argparse
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import gradio as gr
import config


def parse_args():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="ButterVision - Stable Diffusion WebUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Servidor
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host del servidor (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Puerto del servidor (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Crear link público de Gradio (share link)",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        help='Autenticación básica: "usuario:contraseña"',
    )
    
    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"ID del modelo de HuggingFace (default: {config.model_config.model_id})",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="No descargar modelos al iniciar; falla luego si el modelo no está disponible",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Desactivar float16 (usa más VRAM pero puede ser más preciso)",
    )
    parser.add_argument(
        "--no-xformers",
        action="store_true",
        help="Desactivar xformers memory efficient attention",
    )
    
    # Optimizaciones
    parser.add_argument(
        "--lowvram",
        action="store_true",
        help="Modo VRAM extremadamente baja (< 4GB) - activa CPU offload",
    )
    parser.add_argument(
        "--medvram",
        action="store_true",
        help="Modo VRAM media (4-6GB) - solo optimizaciones básicas",
    )
    parser.add_argument(
        "--no-optimizations",
        action="store_true",
        help="Desactivar TODAS las optimizaciones de memoria",
    )
    
    # Otros
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forzar uso de CPU (muy lento)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        choices=["default", "soft", "monochrome"],
        default="default",
        help="Tema de la interfaz",
    )
    
    return parser.parse_args()


def apply_launch_config(args):
    """Aplica configuración desde argumentos de línea de comandos"""
    
    # Configuración del servidor
    config.server_config.host = args.host
    config.server_config.port = args.port
    config.server_config.share = args.share
    
    if args.auth:
        try:
            user, password = args.auth.split(":")
            config.server_config.auth = (user, password)
        except ValueError:
            print("⚠️  Formato de auth inválido. Usa: usuario:contraseña")
    
    # Configuración del modelo
    if args.model:
        config.model_config.model_id = args.model
    
    if args.no_fp16:
        config.model_config.use_fp16 = False
    
    if args.no_xformers:
        config.model_config.enable_xformers = False
    
    # Modos de VRAM
    if args.lowvram:
        print("🔧 Modo LOW VRAM activado (< 4GB)")
        config.model_config.use_fp16 = True
        config.model_config.enable_xformers = True
        config.model_config.enable_attention_slicing = True
        config.model_config.enable_vae_slicing = True
        config.model_config.enable_cpu_offload = True
    
    elif args.medvram:
        print("🔧 Modo MED VRAM activado (4-6GB)")
        config.model_config.use_fp16 = True
        config.model_config.enable_xformers = True
        config.model_config.enable_attention_slicing = True
        config.model_config.enable_vae_slicing = True
        config.model_config.enable_cpu_offload = False
    
    if args.no_optimizations:
        print("⚠️  TODAS las optimizaciones desactivadas")
        config.model_config.use_fp16 = False
        config.model_config.enable_xformers = False
        config.model_config.enable_attention_slicing = False
        config.model_config.enable_vae_slicing = False
        config.model_config.enable_cpu_offload = False
    
    # Tema
    config.ui_config.theme = args.theme


def print_system_info():
    """Imprime información del sistema"""
    print("\n" + "="*60)
    print("🎨 ButterVision - Stable Diffusion WebUI")
    print("="*60)
    
    # PyTorch info
    print(f"\n📦 PyTorch: {torch.__version__}")
    print(f"🔧 CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  No se detectó GPU CUDA - usando CPU (muy lento)")
    
    # Configuración
    print(f"\n📝 Modelo: {config.model_config.model_id}")
    print(f"🔧 Float16: {config.model_config.use_fp16}")
    print(f"⚡ xformers: {config.model_config.enable_xformers}")
    print(f"✂️  Attention slicing: {config.model_config.enable_attention_slicing}")
    print(f"🔪 VAE slicing: {config.model_config.enable_vae_slicing}")
    print(f"💻 CPU offload: {config.model_config.enable_cpu_offload}")
    
    print(f"\n🌐 Servidor: {config.server_config.host}:{config.server_config.port}")
    print(f"🔗 Share link: {config.server_config.share}")
    
    print("\n" + "="*60 + "\n")


def bootstrap_base_models(skip_download: bool = False):
    """Descarga o resuelve los modelos base mínimos antes de abrir la UI."""
    from core.model_manager import ModelManager

    model_id = config.model_config.model_id
    print("📦 Verificando modelos base iniciales...")

    model_manager = ModelManager()
    model_path = model_manager.ensure_model(
        model_id=model_id,
        allow_download=not skip_download,
    )

    config.model_config.model_id = model_path
    print(f"✅ Modelo activo: {config.model_config.model_id}\n")


def main():
    """Función principal"""
    
    # Parsear argumentos
    args = parse_args()
    
    # Aplicar configuración
    apply_launch_config(args)
    
    # Mostrar info del sistema
    print_system_info()

    # Garantizar modelo base inicial
    try:
        bootstrap_base_models(skip_download=args.skip_model_download)
    except Exception as e:
        print(f"❌ No se pudo preparar el modelo base: {e}")
        print("   Revisa tu conexión o ejecuta con un modelo local válido usando --model /ruta/modelo")
        sys.exit(1)
    
    # Crear interfaz
    from ui import create_ui
    from ui.interface import get_ui_css, get_ui_js

    print("🚀 Iniciando interfaz web...\n")
    interface = create_ui()
    
    # Preparar kwargs para launch
    launch_kwargs = {
        "server_name": config.server_config.host,
        "server_port": config.server_config.port,
        "share": config.server_config.share,
        "inbrowser": False,  # No abrir navegador automáticamente
        "show_error": config.ui_config.show_error,
        "theme": gr.themes.Soft(),
        "footer_links": [],
    }
    
    # Añadir autenticación si está configurada
    if config.server_config.auth:
        launch_kwargs["auth"] = config.server_config.auth
    
    # Lanzar
    try:
        interface.launch(**launch_kwargs)
    except KeyboardInterrupt:
        print("\n\n👋 Cerrando ButterVision...")
    except Exception as e:
        print(f"\n❌ Error al iniciar: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
