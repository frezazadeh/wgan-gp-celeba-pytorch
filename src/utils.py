import os
import torch
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from src.config import WANDB_ENABLED, CHECKPOINT_DIR

def initialize_weights(m):
    """Initializes weights for Conv2d, ConvTranspose2d, and BatchNorm2d layers."""
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def show_tensor_images(image_tensor, num: int = 25, name: str = 'image'):
    """Visualizes a tensor of images and optionally logs to W&B."""
    # Denormalize from [-1, 1] to [0, 1] for display
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num], nrow=5)
    image_grid = (image_grid + 1) / 2 # Denormalize
    
    if WANDB_ENABLED and wandb.run:
        wandb.log({name: wandb.Image(image_grid.permute(1, 2, 0).numpy())})
        
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

def save_checkpoint(filename: str, gen, crit, gen_opt, crit_opt):
    """Saves model and optimizer states to a checkpoint file."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    g_path = os.path.join(CHECKPOINT_DIR, f"G_{filename}")
    c_path = os.path.join(CHECKPOINT_DIR, f"C_{filename}")
    
    torch.save({
        'model_state_dict': gen.state_dict(),
        'opt_state_dict': gen_opt.state_dict(),
    }, g_path)
    
    torch.save({
        'model_state_dict': crit.state_dict(),
        'opt_state_dict': crit_opt.state_dict(),
    }, c_path)
    
    print(f"[INFO] Saved checkpoint: {filename}")

def load_checkpoint(filename: str, gen, crit, gen_opt, crit_opt):
    """Loads model and optimizer states from a checkpoint file."""
    g_path = os.path.join(CHECKPOINT_DIR, f"G_{filename}")
    c_path = os.path.join(CHECKPOINT_DIR, f"C_{filename}")

    try:
        g_ckpt = torch.load(g_path)
        c_ckpt = torch.load(c_path)
        
        gen.load_state_dict(g_ckpt['model_state_dict'])
        gen_opt.load_state_dict(g_ckpt['opt_state_dict'])
        crit.load_state_dict(c_ckpt['model_state_dict'])
        crit_opt.load_state_dict(c_ckpt['opt_state_dict'])
        
        print(f"[INFO] Loaded checkpoint: {filename}")
    except FileNotFoundError:
        print(f"[WARNING] Checkpoint file not found: {filename}. Starting from scratch.")
