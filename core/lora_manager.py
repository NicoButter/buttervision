"""
LoRA Manager - Gestión de LoRAs dinámicos
Permite cargar, descargar y aplicar LoRAs a los pipelines
"""
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional
from safetensors.torch import load_file
import config


class LoRAManager:
    """
    Gestor de LoRAs (Low-Rank Adaptation)
    Permite aplicar múltiples LoRAs simultáneamente al modelo
    """
    
    def __init__(self, lora_dir: Path = None):
        """
        Inicializa el gestor de LoRAs
        
        Args:
            lora_dir: Directorio donde están los archivos .safetensors de LoRAs
        """
        self.lora_dir = lora_dir or config.LORA_DIR
        self.loaded_loras: Dict[str, float] = {}  # {nombre: peso}
        self.lora_paths: Dict[str, Path] = {}  # {nombre: ruta_completa}
        
        # Escanear directorio de LoRAs
        self.scan_lora_directory()
    
    def scan_lora_directory(self) -> List[str]:
        """
        Escanea el directorio de LoRAs y retorna los nombres disponibles
        
        Returns:
            Lista de nombres de LoRAs disponibles
        """
        self.lora_paths = {}
        
        if not self.lora_dir.exists():
            print(f"⚠️  Directorio de LoRAs no existe: {self.lora_dir}")
            return []
        
        # Buscar archivos .safetensors y .pt
        for extension in ["*.safetensors", "*.pt", "*.bin"]:
            for lora_file in self.lora_dir.glob(extension):
                lora_name = lora_file.stem  # Nombre sin extensión
                self.lora_paths[lora_name] = lora_file
        
        if self.lora_paths:
            print(f"📚 LoRAs encontrados: {len(self.lora_paths)}")
            for name in self.lora_paths.keys():
                print(f"   - {name}")
        else:
            print(f"ℹ️  No hay LoRAs en {self.lora_dir}")
        
        return list(self.lora_paths.keys())
    
    def get_available_loras(self) -> List[str]:
        """Retorna lista de LoRAs disponibles"""
        return list(self.lora_paths.keys())
    
    def load_lora(self, lora_name: str, weight: float = 1.0, pipeline=None) -> bool:
        """
        Carga un LoRA en el pipeline
        
        Args:
            lora_name: Nombre del LoRA (sin extensión)
            weight: Peso del LoRA (0.0 a 2.0, típico 0.5-1.0)
            pipeline: Pipeline donde aplicar el LoRA
        
        Returns:
            True si se cargó exitosamente
        """
        if lora_name not in self.lora_paths:
            print(f"❌ LoRA '{lora_name}' no encontrado")
            return False
        
        if pipeline is None:
            print("⚠️  No hay pipeline activo para cargar el LoRA")
            return False
        
        try:
            lora_path = self.lora_paths[lora_name]
            print(f"📦 Cargando LoRA: {lora_name} (peso: {weight})")
            
            # Usar el método load_lora_weights de diffusers
            pipeline.load_lora_weights(
                str(lora_path.parent),
                weight_name=lora_path.name,
                adapter_name=lora_name,
            )
            
            # Configurar peso del adaptador
            pipeline.set_adapters([lora_name], adapter_weights=[weight])
            
            self.loaded_loras[lora_name] = weight
            print(f"✅ LoRA '{lora_name}' cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar LoRA '{lora_name}': {e}")
            return False
    
    def unload_lora(self, lora_name: str, pipeline=None) -> bool:
        """
        Descarga un LoRA del pipeline
        
        Args:
            lora_name: Nombre del LoRA a descargar
            pipeline: Pipeline del cual remover el LoRA
        
        Returns:
            True si se descargó exitosamente
        """
        if lora_name not in self.loaded_loras:
            print(f"ℹ️  LoRA '{lora_name}' no está cargado")
            return False
        
        if pipeline is None:
            print("⚠️  No hay pipeline activo")
            return False
        
        try:
            # Remover el adaptador
            pipeline.delete_adapters(lora_name)
            del self.loaded_loras[lora_name]
            print(f"♻️  LoRA '{lora_name}' descargado")
            return True
        except Exception as e:
            print(f"❌ Error al descargar LoRA: {e}")
            return False
    
    def unload_all_loras(self, pipeline=None):
        """Descarga todos los LoRAs activos"""
        if not self.loaded_loras:
            print("ℹ️  No hay LoRAs cargados")
            return
        
        lora_names = list(self.loaded_loras.keys())
        for lora_name in lora_names:
            self.unload_lora(lora_name, pipeline)
        
        print("♻️  Todos los LoRAs descargados")
    
    def update_lora_weight(self, lora_name: str, weight: float, pipeline=None) -> bool:
        """
        Actualiza el peso de un LoRA ya cargado
        
        Args:
            lora_name: Nombre del LoRA
            weight: Nuevo peso (0.0 a 2.0)
            pipeline: Pipeline activo
        
        Returns:
            True si se actualizó exitosamente
        """
        if lora_name not in self.loaded_loras:
            print(f"⚠️  LoRA '{lora_name}' no está cargado")
            return False
        
        if pipeline is None:
            return False
        
        try:
            # Actualizar peso del adaptador
            pipeline.set_adapters([lora_name], adapter_weights=[weight])
            self.loaded_loras[lora_name] = weight
            print(f"🔄 Peso de '{lora_name}' actualizado a {weight}")
            return True
        except Exception as e:
            print(f"❌ Error al actualizar peso: {e}")
            return False
    
    def get_loaded_loras(self) -> Dict[str, float]:
        """Retorna diccionario de LoRAs cargados con sus pesos"""
        return self.loaded_loras.copy()
    
    def load_multiple_loras(
        self,
        lora_configs: List[tuple],  # [(nombre, peso), ...]
        pipeline=None
    ) -> bool:
        """
        Carga múltiples LoRAs simultáneamente
        
        Args:
            lora_configs: Lista de tuplas (nombre, peso)
            pipeline: Pipeline donde aplicar
        
        Returns:
            True si todos se cargaron exitosamente
        """
        if not lora_configs:
            return True
        
        success = True
        for lora_name, weight in lora_configs:
            if not self.load_lora(lora_name, weight, pipeline):
                success = False
        
        return success
    
    def get_lora_info(self, lora_name: str) -> Optional[dict]:
        """
        Obtiene información sobre un LoRA
        
        Args:
            lora_name: Nombre del LoRA
        
        Returns:
            Diccionario con información o None
        """
        if lora_name not in self.lora_paths:
            return None
        
        lora_path = self.lora_paths[lora_name]
        
        info = {
            "name": lora_name,
            "path": str(lora_path),
            "size_mb": lora_path.stat().st_size / (1024 * 1024),
            "loaded": lora_name in self.loaded_loras,
            "weight": self.loaded_loras.get(lora_name, 0.0),
        }
        
        return info


# Instancia global del manager
lora_manager = LoRAManager()
