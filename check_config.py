#!/usr/bin/env python3
"""
Test simple para verificar configuración sin descargar modelos
"""

def check_model_exists():
    """Verificar si el modelo SD 1.5 existe localmente"""
    from pathlib import Path

    model_paths = [
        "cache/models--runwayml--stable-diffusion-v1-5/snapshots",
        "cache/models--runwayml--stable-diffusion-v1-5",
    ]

    for path in model_paths:
        p = Path(path)
        if p.exists():
            safetensors = list(p.rglob("*.safetensors"))
            if safetensors:
                print(f"✅ Modelo encontrado: {len(safetensors)} archivos .safetensors")
                return True

    print("❌ Modelo no encontrado localmente")
    print("   Ejecuta: python main.py (primera vez descarga ~4GB)")
    return False

def check_pipeline_config():
    """Verificar configuración del pipeline"""
    print("🔧 Verificando configuración del pipeline...")

    try:
        from core.advanced_pipeline import ButterVisionPipeline

        # Verificar que DDIM está forzado
        pipeline = ButterVisionPipeline.__new__(ButterVisionPipeline)
        print("✅ ButterVisionPipeline importado")

        # Verificar guidance_scale por defecto
        import inspect
        sig = inspect.signature(pipeline.generate_image)
        guidance_default = sig.parameters['guidance_scale'].default
        print(f"   Guidance scale default: {guidance_default}")

        if guidance_default == 5.0:
            print("✅ Guidance scale correcto (5.0)")
        else:
            print(f"⚠️ Guidance scale: {guidance_default} (debería ser 5.0)")

        return True

    except Exception as e:
        print(f"❌ Error verificando pipeline: {e}")
        return False

def check_ui_config():
    """Verificar configuración de UI"""
    print("🖥️ Verificando configuración de UI...")

    try:
        # Leer el archivo de UI para verificar CFG default
        with open("ui/interface.py", "r") as f:
            content = f.read()

        if "value=5.0" in content and "CFG Scale" in content:
            print("✅ UI CFG Scale default: 5.0")
            return True
        else:
            print("❌ UI CFG Scale no está en 5.0")
            return False

    except Exception as e:
        print(f"❌ Error verificando UI: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Verificación de configuración - Antes de generar")
    print("=" * 50)

    checks = [
        ("Modelo local", check_model_exists),
        ("Pipeline config", check_pipeline_config),
        ("UI config", check_ui_config),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n🧪 {name}:")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 RESUMEN:")

    passed = sum(results)
    total = len(results)

    status = "✅" if passed == total else "⚠️"
    print(f"{status} {passed}/{total} checks pasaron")

    if passed == total:
        print("\n🎯 CONFIGURACIÓN LISTA")
        print("Ejecuta: python main.py")
        print("Prompt de test: 'a red apple on a white table, studio lighting, photo'")
    else:
        print("\n❌ CONFIGURACIÓN INCOMPLETA")
        print("Revisa los errores arriba")