import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.models.ddpm_model import DDPMModel

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir", type=str, default="ddpm_training",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Total number of training steps.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    return parser.parse_args()


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model and scheduler
    unet = ...  # Initialize UNet model
    scheduler = DDPMScheduler()
    model = DDPMModel(unet=unet, scheduler=scheduler)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = ...  # Load your dataset here
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Training loop
    for step in range(args.max_train_steps):
        for batch in dataloader:
            # Forward diffusion
            x_0 = batch["image"].to(accelerator.device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (x_0.size(0),), device=accelerator.device)
            loss = model.forward_diffusion(x_0, timesteps)

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                logger.info(f"Step {step}: loss = {loss.item()}")

            if step >= args.max_train_steps:
                break

    # Save the model
    if accelerator.is_main_process:
        model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)