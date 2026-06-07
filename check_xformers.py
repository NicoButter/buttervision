#!/usr/bin/env python3
"""
Verificar que xformers está funcionando correctamente con CUDA
"""

import torch
import sys

def check_xformers():
    print("🔍 Verificando xformers con CUDA...")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA no disponible")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA versión: {torch.version.cuda}")

    try:
        import xformers
        print(f"✅ xformers importado: {xformers.__version__}")

        # Verificar que tiene ops CUDA
        from xformers.ops import memory_efficient_attention
        print("✅ memory_efficient_attention disponible")

        # Probar una operación simple
        device = torch.device("cuda")
        q = torch.randn(1, 32, 64, 64, device=device)
        k = torch.randn(1, 32, 64, 64, device=device)
        v = torch.randn(1, 32, 64, 64, device=device)

        with torch.no_grad():
            out = memory_efficient_attention(q, k, v)
            print("✅ Operación CUDA exitosa")
            print(f"Output shape: {out.shape}")

        # Verificar VRAM usado
        vram_used = torch.cuda.memory_allocated() / 1024**3
        print(f"VRAM usada: {vram_used:.2f} GB")

        return True

    except ImportError as e:
        print(f"❌ Error importando xformers: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en operación CUDA: {e}")
        return False

if __name__ == "__main__":
    success = check_xformers()
    sys.exit(0 if success else 1)
