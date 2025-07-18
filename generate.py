import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from src.models import Generator
from src.config import Z_DIM, DEVICE

def generate_images(checkpoint_path: str, num_images: int = 25):
    """Generates and displays a grid of images from a trained generator."""
    print(f"[INFO] Loading generator from checkpoint: {checkpoint_path}")
    gen = Generator(z_dim=Z_DIM).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    gen.load_state_dict(checkpoint['model_state_dict'])
    gen.eval()

    noise = torch.randn(num_images, Z_DIM, device=DEVICE)
    
    with torch.no_grad():
        fake_images = gen(noise).detach().cpu()

    # Visualize the generated images
    from src.utils import show_tensor_images
    print("[INFO] Generated Images:")
    show_tensor_images(fake_images, num=num_images)

def generate_latent_space_morph(checkpoint_path: str, rows: int = 4, steps: int = 17):
    """Generates and displays a latent space interpolation (morphing)."""
    print(f"[INFO] Loading generator from checkpoint: {checkpoint_path}")
    gen = Generator(z_dim=Z_DIM).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    gen.load_state_dict(checkpoint['model_state_dict'])
    gen.eval()

    gen_set = []
    print("[INFO] Creating latent space interpolation...")
    for _ in range(rows):
        # Create start and end points in the latent space
        z1 = torch.randn(1, Z_DIM, 1, 1, device=DEVICE)
        z2 = torch.randn(1, Z_DIM, 1, 1, device=DEVICE)
        
        # Linearly interpolate between the two points
        for alpha in np.linspace(0, 1, steps):
            z = alpha * z1 + (1 - alpha) * z2
            with torch.no_grad():
                img_tensor = gen(z.view(1, Z_DIM, 1, 1))[0]
            # Permute and clip the image for display
            img_display = img_tensor.cpu().detach().permute(1, 2, 0).clip(0, 1)
            gen_set.append(img_display)
            
    # Create and display the grid
    fig = plt.figure(figsize=(steps * 1.5, rows * 1.5))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, steps), axes_pad=0.1)
    
    for ax, img in zip(grid, gen_set):
        ax.axis('off')
        ax.imshow(img)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images from a trained WGAN-GP model.")
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help="Path to the generator checkpoint file (e.g., 'checkpoints/G_step_105.pth')."
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint file not found at: {args.checkpoint}")
    else:
        generate_images(args.checkpoint)
        generate_latent_space_morph(args.checkpoint)
