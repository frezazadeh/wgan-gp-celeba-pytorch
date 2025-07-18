import torch
from tqdm.auto import tqdm
import wandb

from src.config import (
    N_EPOCHS, DEVICE, Z_DIM, CRITIC_CYCLES, GRADIENT_PENALTY_LAMBDA,
    WANDB_ENABLED, WANDB_PROJECT_NAME, SHOW_STEP, SAVE_STEP, BATCH_SIZE
)
from src.utils import show_tensor_images, save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, generator, critic, gen_optimizer, crit_optimizer, dataloader):
        self.gen = generator
        self.crit = critic
        self.gen_opt = gen_optimizer
        self.crit_opt = crit_optimizer
        self.dataloader = dataloader
        
        self.cur_step = 0
        self.gen_losses = []
        self.crit_losses = []

        if WANDB_ENABLED:
            self._init_wandb()

    def _init_wandb(self):
        """Initializes Weights & Biases for experiment tracking."""
        wandb.init(
            project=WANDB_PROJECT_NAME,
            config={
                "epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "z_dim": Z_DIM,
                "critic_cycles": CRITIC_CYCLES,
                "gradient_penalty": GRADIENT_PENALTY_LAMBDA,
            }
        )
        wandb.watch(self.gen, log="all", log_freq=100)
        wandb.watch(self.crit, log="all", log_freq=100)

    def _compute_gradient_penalty(self, real, fake):
        """Calculates the gradient penalty term for WGAN-GP."""
        alpha = torch.randn(real.size(0), 1, 1, 1, device=DEVICE)
        mixed_images = (real * alpha + fake * (1 - alpha)).requires_grad_(True)
        mixed_scores = self.crit(mixed_images)
        
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        gp = GRADIENT_PENALTY_LAMBDA * ((gradient_norm - 1) ** 2).mean()
        return gp

    def _train_critic_step(self, real_images):
        """Performs a single training step for the critic."""
        batch_size = real_images.size(0)
        mean_iteration_critic_loss = 0

        for _ in range(CRITIC_CYCLES):
            self.crit_opt.zero_grad()
            noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_images = self.gen(noise).detach()
            
            real_pred = self.crit(real_images)
            fake_pred = self.crit(fake_images)
            
            gp = self._compute_gradient_penalty(real_images, fake_images)
            
            # Wasserstein-GP loss for the critic
            loss_c = fake_pred.mean() - real_pred.mean() + gp
            
            loss_c.backward()
            self.crit_opt.step()
            
            mean_iteration_critic_loss += loss_c.item() / CRITIC_CYCLES
            
        return mean_iteration_critic_loss

    def _train_generator_step(self, batch_size):
        """Performs a single training step for the generator."""
        self.gen_opt.zero_grad()
        noise = torch.randn(batch_size, Z_DIM, device=DEVICE)
        fake_images = self.gen(noise)
        
        fake_pred = self.crit(fake_images)
        
        # Generator loss is to maximize the critic's output for fake images
        loss_g = -fake_pred.mean()
        
        loss_g.backward()
        self.gen_opt.step()
        
        return loss_g.item()

    def train(self):
        """Main training loop."""
        for epoch in range(N_EPOCHS):
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
            
            for real_images in progress_bar:
                real_images = real_images.to(DEVICE)
                batch_size = real_images.size(0)
                
                # --- Train Critic ---
                crit_loss = self._train_critic_step(real_images)
                self.crit_losses.append(crit_loss)
                
                # --- Train Generator ---
                gen_loss = self._train_generator_step(batch_size)
                self.gen_losses.append(gen_loss)

                # --- Logging and Visualization ---
                if WANDB_ENABLED:
                    wandb.log({
                        "critic_loss": crit_loss,
                        "generator_loss": gen_loss,
                        "step": self.cur_step
                    })
                
                progress_bar.set_postfix({
                    "crit_loss": f"{crit_loss:.4f}",
                    "gen_loss": f"{gen_loss:.4f}"
                })

                if self.cur_step > 0 and self.cur_step % SHOW_STEP == 0:
                    print(f"\n--- Step {self.cur_step}: Visualizing Results ---")
                    noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
                    fake = self.gen(noise)
                    show_tensor_images(fake, name="Generated Images")
                    show_tensor_images(real_images, name="Real Images")
                
                if self.cur_step > 0 and self.cur_step % SAVE_STEP == 0:
                    save_checkpoint(f"step_{self.cur_step}.pth", self.gen, self.crit, self.gen_opt, self.crit_opt)

                self.cur_step += 1
        
        if WANDB_ENABLED:
            wandb.finish()

    def load_checkpoint(self, filename: str):
        """Helper method to load a checkpoint for the trainer."""
        load_checkpoint(filename, self.gen, self.crit, self.gen_opt, self.crit_opt)
