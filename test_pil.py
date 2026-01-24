#!/usr/bin/env python3
"""
Test mínimo para verificar que PIL Image funciona correctamente con Gradio
"""

from PIL import Image, ImageDraw
import numpy as np

def create_test_image():
    """Crear una imagen de test simple"""
    # Crear imagen RGB
    img = Image.new('RGB', (512, 512), color='red')

    # Dibujar algo
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill='blue')
    draw.text((200, 200), "Test Image", fill='white')

    return img

def test_pil_image():
    """Test que PIL Image se guarda correctamente"""
    print("🧪 Creando imagen de test...")

    img = create_test_image()

    # Verificar tipo
    print(f"Tipo de imagen: {type(img)}")
    print(f"Modo: {img.mode}")
    print(f"Tamaño: {img.size}")

    # Guardar como PNG
    img.save("test_pil.png")
    print("✅ Imagen guardada como test_pil.png")

    # Verificar que no es numpy
    try:
        arr = np.array(img)
        print(f"⚠️ Se puede convertir a numpy: {arr.shape}")
    except Exception as e:
        print(f"❌ Error convirtiendo a numpy: {e}")

    return img

if __name__ == "__main__":
    test_pil_image()