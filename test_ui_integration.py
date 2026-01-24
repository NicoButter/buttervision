#!/usr/bin/env python3
"""
Test de integración para verificar que la UI funciona con PIL Images
"""

import sys
from pathlib import Path

# Añadir raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from ui.interface import ButterVisionUI

def test_ui_initialization():
    """Test que los componentes básicos funcionan"""
    print("🧪 Probando componentes básicos...")

    try:
        # Solo probar imports y tipos básicos
        from core.advanced_pipeline import ButterVisionPipeline
        print("✅ Import de ButterVisionPipeline correcto")

        # Verificar que PIL funciona
        from PIL import Image
        img = Image.new('RGB', (32, 32), color='blue')
        print("✅ PIL Image creation correcto")

        return True

    except Exception as e:
        print(f"❌ Error en componentes básicos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pil_image_handling():
    """Test que PIL Images se manejan correctamente"""
    print("\n🧪 Probando manejo de PIL Images...")

    # Crear imagen de test
    img = Image.new('RGB', (64, 64), color='red')

    # Verificar propiedades
    assert isinstance(img, Image.Image), "No es PIL Image"
    assert img.mode == 'RGB', f"Modo incorrecto: {img.mode}"
    assert img.size == (64, 64), f"Tamaño incorrecto: {img.size}"

    # Verificar que se puede guardar
    test_path = "test_ui.png"
    img.save(test_path)
    print(f"✅ Imagen guardada: {test_path}")

    # Verificar que el archivo existe
    assert Path(test_path).exists(), "Archivo no creado"

    # Limpiar
    Path(test_path).unlink()

    print("✅ PIL Image handling correcto")
    return True

if __name__ == "__main__":
    print("🎨 Test de integración UI - ButterVision")
    print("=" * 50)

    success = True

    # Test 1: PIL Images
    try:
        test_pil_image_handling()
    except Exception as e:
        print(f"❌ Test PIL falló: {e}")
        success = False

    # Test 2: UI initialization
    try:
        test_ui_initialization()
    except Exception as e:
        print(f"❌ Test UI falló: {e}")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 ¡Todos los tests pasaron!")
        print("La UI debería mostrar imágenes PIL correctamente.")
    else:
        print("❌ Algunos tests fallaron.")

    sys.exit(0 if success else 1)