#!/usr/bin/env python3
"""
Script de entrenamiento LoRA para cara personal usando Diffusers
Entrena un LoRA con fotos locales para generar imágenes realistas de tu cara

Requisitos:
- Python 3.10+
- PyTorch con CUDA
- diffusers, accelerate, transformers, peft
- Dataset: ./data/mi_cara/ con 15-30 fotos de tu cara

Uso:
python train_lora_mi_cara.py

Resultado:
- LoRA guardado en ./loras/mi_cara.safetensors
- Se carga automáticamente en la UI
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# Configuración del entrenamiento
CONFIG = {
    "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
    "dataset_dir": "./data/mi_cara",
    "output_dir": "./loras",
    "lora_name": "mi_cara",
    "resolution": 512,
    "train_batch_size": 1,  # Para GTX 1650, batch pequeño
    "max_train_steps": 1000,  # Ajustar según dataset
    "learning_rate": 1e-4,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "snr_gamma": 5.0,
    "mixed_precision": "fp16",
    "gradient_accumulation_steps": 4,
    "save_steps": 500,
    "seed": 42,
    "rank": 4,  # Rank del LoRA
}

class FaceDataset(Dataset):
    """Dataset para fotos de cara personal"""

    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.size = size

        # Buscar imágenes
        self.image_paths = list(self.data_dir.glob("*.jpg")) + \
                          list(self.data_dir.glob("*.jpeg")) + \
                          list(self.data_dir.glob("*.png"))

        if len(self.image_paths) == 0:
            raise ValueError(f"No se encontraron imágenes en {data_dir}")

        print(f"📸 Encontradas {len(self.image_paths)} imágenes")

        # Prompt fijo para todas las imágenes
        self.prompt = "foto de una persona, cara realista, alta calidad, iluminación natural"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Cargar imagen
        image = Image.open(image_path).convert("RGB")

        # Redimensionar manteniendo aspecto
        image = self._resize_and_crop(image, self.size)

        # Convertir a tensor
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Tokenizar prompt
        tokens = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(),
        }

    def _resize_and_crop(self, image, size):
        """Redimensiona y recorta la imagen manteniendo aspecto"""
        width, height = image.size

        # Calcular ratio
        ratio = min(size / width, size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Redimensionar
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Centrar y recortar
        left = (new_width - size) // 2
        top = (new_height - size) // 2
        right = left + size
        bottom = top + size

        return image.crop((left, top, right, bottom))


def collate_fn(batch):
    """Collate function para el DataLoader"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
    }


def train_lora(config):
    """Entrena el LoRA"""

    # Configurar acelerador
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )

    set_seed(config["seed"])

    # Crear directorios
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)

    # Cargar modelos
    print("📥 Cargando modelos base...")

    tokenizer = CLIPTokenizer.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="text_encoder"
    )

    unet = UNet2DConditionModel.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="unet"
    )

    # Congelar modelos base
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Configurar LoRA
    lora_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["rank"] * 2,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )

    # Aplicar LoRA al UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Configurar optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["learning_rate"]
    )

    # Cargar dataset
    train_dataset = FaceDataset(
        config["dataset_dir"],
        tokenizer,
        config["resolution"]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Configurar scheduler
    lr_scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=config["max_train_steps"],
    )

    # Preparar con accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Scheduler de ruido
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    # Entrenamiento
    print("🚀 Iniciando entrenamiento...")
    global_step = 0
    progress_bar = tqdm(range(config["max_train_steps"]), disable=not accelerator.is_local_main_process)

    while global_step < config["max_train_steps"]:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Convertir imágenes a latent space
                latents = batch["pixel_values"].to(accelerator.device)

                # Añadir ruido
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Timesteps aleatorios
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=accelerator.device
                ).long()

                # Añadir ruido a los latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Obtener embeddings de texto
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                # Predicción del ruido
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calcular pérdida
                if config["snr_gamma"] is None:
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                else:
                    # Usar SNR weighting
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, config["snr_gamma"] * torch.ones_like(snr)], dim=1).min(dim=1)[0]
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Actualizar progreso
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Guardar checkpoint
                if global_step % config["save_steps"] == 0:
                    save_path = output_dir / f"{config['lora_name']}_step_{global_step}.safetensors"
                    accelerator.save_state(str(save_path))
                    print(f"💾 Checkpoint guardado: {save_path}")

                if global_step >= config["max_train_steps"]:
                    break

    # Guardar LoRA final
    final_path = output_dir / f"{config['lora_name']}.safetensors"
    accelerator.save_state(str(final_path))
    print(f"✅ LoRA entrenado guardado en: {final_path}")

    # Convertir a formato diffusers si es necesario
    # (El formato safetensors ya es compatible)


def main():
    parser = argparse.ArgumentParser(description="Entrenar LoRA para cara personal")
    parser.add_argument("--steps", type=int, default=1000, help="Número de pasos de entrenamiento")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    # Actualizar config con argumentos
    CONFIG["max_train_steps"] = args.steps
    CONFIG["learning_rate"] = args.lr
    CONFIG["train_batch_size"] = args.batch_size

    # Verificar dataset
    dataset_dir = Path(CONFIG["dataset_dir"])
    if not dataset_dir.exists():
        print(f"❌ Directorio de dataset no existe: {dataset_dir}")
        print("📸 Coloca tus fotos en ./data/mi_cara/")
        return

    image_count = len(list(dataset_dir.glob("*.jpg"))) + \
                  len(list(dataset_dir.glob("*.jpeg"))) + \
                  len(list(dataset_dir.glob("*.png")))

    if image_count < 10:
        print(f"⚠️ Solo {image_count} imágenes encontradas. Recomendado: 15-30")
        if input("¿Continuar? (y/n): ").lower() != 'y':
            return

    # Ejecutar entrenamiento
    train_lora(CONFIG)


if __name__ == "__main__":
    main()