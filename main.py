import torch
import torch.optim as optim
from src.config import (DEVICE, Z_DIM, LEARNING_RATE, BETA_1, BETA_2, WANDB_ENABLED)
from src.dataset import get_celeba_dataloader
from src.models import Generator, Critic
from src.trainer import Trainer
from src.utils import initialize_weights

def run_training():
    """
    Initializes models, data, and trainer, then starts the training process.
    """
    print(f"[INFO] Using device: {DEVICE}")

    # Initialize models
    generator = Generator(z_dim=Z_DIM).to(DEVICE)
    critic = Critic().to(DEVICE)

    # Initialize weights
    generator.apply(initialize_weights)
    critic.apply(initialize_weights)

    # Initialize optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
    crit_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

    # Prepare DataLoader
    dataloader = get_celeba_dataloader()

    # Initialize the Trainer
    wgan_trainer = Trainer(
        generator=generator,
        critic=critic,
        gen_optimizer=gen_optimizer,
        crit_optimizer=crit_optimizer,
        dataloader=dataloader,
    )
    
    # Optionally load a checkpoint to resume training
    # wgan_trainer.load_checkpoint(filename="step_35.pth")

    # Start training
    print("[INFO] Starting training...")
    wgan_trainer.train()
    print("[INFO] Training complete.")

if __name__ == "__main__":
    if WANDB_ENABLED:
        import wandb
        # Log in to W&B (you will be prompted for your API key)
        wandb.login()
        
    run_training()
