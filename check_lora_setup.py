#!/usr/bin/env python3
"""
Verificación de configuración LoRA para ButterVision
Comprueba que todo esté listo para entrenar y usar LoRA de cara personal
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Requiere 3.10+")
        return False

def check_directories():
    """Verificar directorios necesarios"""
    print("\n📁 Verificando directorios...")

    dirs_to_check = [
        ("./data/mi_cara", "Dataset de fotos"),
        ("./loras", "LoRAs entrenados"),
        ("./models", "Modelos base"),
        ("./cache", "Cache de modelos"),
    ]

    all_ok = True
    for dir_path, description in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path} - {description}")
        else:
            print(f"❌ {dir_path} - {description} (no existe)")
            all_ok = False

    return all_ok

def check_dataset():
    """Verificar dataset de fotos"""
    print("\n📸 Verificando dataset...")

    dataset_dir = Path("./data/mi_cara")
    if not dataset_dir.exists():
        print("❌ Directorio ./data/mi_cara no existe")
        return False

    # Buscar imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_count = 0

    for ext in image_extensions:
        image_count += len(list(dataset_dir.glob(ext)))

    if image_count == 0:
        print("❌ No se encontraron imágenes en ./data/mi_cara")
        print("   Formatos soportados: .jpg, .jpeg, .png")
        return False
    elif image_count < 10:
        print(f"⚠️ Solo {image_count} imágenes encontradas (recomendado: 15-30)")
    else:
        print(f"✅ {image_count} imágenes encontradas")

    return True

def check_dependencies():
    """Verificar dependencias de Python"""
    print("\n📦 Verificando dependencias...")

    required_packages = [
        'torch',
        'diffusers',
        'transformers',
        'accelerate',
        'peft',
        'tqdm',
        'PIL',
        'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Paquetes faltantes: {', '.join(missing_packages)}")
        print("Instala con: pip install " + " ".join(missing_packages))
        return False

    return True

def check_cuda():
    """Verificar CUDA"""
    print("\n🖥️ Verificando CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✅ CUDA disponible: {device_name} ({device_count} GPU(s))")
            return True
        else:
            print("❌ CUDA no disponible")
            return False
    except ImportError:
        print("❌ PyTorch no instalado")
        return False

def check_lora_file():
    """Verificar si existe LoRA entrenado"""
    print("\n🎭 Verificando LoRA entrenado...")

    lora_path = Path("./loras/mi_cara.safetensors")
    if lora_path.exists():
        size_mb = lora_path.stat().st_size / (1024 * 1024)
        print(f"✅ LoRA encontrado: {lora_path} ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠️ LoRA no encontrado: ./loras/mi_cara.safetensors")
        print("   Entrena uno con: python train_lora_mi_cara.py")
        return False

def check_training_script():
    """Verificar script de entrenamiento"""
    print("\n🚀 Verificando script de entrenamiento...")

    script_path = Path("train_lora_mi_cara.py")
    if script_path.exists():
        print(f"✅ Script encontrado: {script_path}")
        return True
    else:
        print(f"❌ Script no encontrado: {script_path}")
        return False

def main():
    """Función principal"""
    print("🔍 Verificación de configuración LoRA - ButterVision")
    print("=" * 50)

    checks = [
        ("Python", check_python_version),
        ("Directorios", check_directories),
        ("Dataset", check_dataset),
        ("Dependencias", check_dependencies),
        ("CUDA", check_cuda),
        ("Script de entrenamiento", check_training_script),
        ("LoRA entrenado", check_lora_file),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
            results.append((name, False))

    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN:")

    passed = 0
    total = len(results)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
        if result:
            passed += 1

    print(f"\n{passed}/{total} checks pasaron")

    if passed == total:
        print("🎉 ¡Todo listo para usar LoRA!")
        print("\nPara entrenar:")
        print("   python train_lora_mi_cara.py")
        print("\nPara generar imágenes:")
        print("   python main.py")
    elif passed >= total - 1:  # Solo falta el LoRA
        print("⚠️ Casi listo. Solo falta entrenar el LoRA:")
        print("   python train_lora_mi_cara.py")
    else:
        print("❌ Configuración incompleta. Revisa los errores arriba.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)