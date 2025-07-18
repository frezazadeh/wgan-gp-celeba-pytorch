import torch.nn as nn
from src.config import GENERATOR_DIM, CRITIC_DIM

class Generator(nn.Module):
    def __init__(self, z_dim: int, d_dim: int = GENERATOR_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self._make_gen_block(z_dim, d_dim * 32, 4, 1, 0),
            self._make_gen_block(d_dim * 32, d_dim * 16, 4, 2, 1),
            self._make_gen_block(d_dim * 16, d_dim * 8, 4, 2, 1),
            self._make_gen_block(d_dim * 8, d_dim * 4, 4, 2, 1),
            self._make_gen_block(d_dim * 4, d_dim * 2, 4, 2, 1),
            # Final layer
            nn.ConvTranspose2d(d_dim * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def _make_gen_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, noise):
        # Reshape noise to be compatible with ConvTranspose2d
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

class Critic(nn.Module):
    def __init__(self, d_dim: int = CRITIC_DIM):
        super().__init__()
        self.crit = nn.Sequential(
            self._make_crit_block(3, d_dim, 4, 2, 1),
            self._make_crit_block(d_dim, d_dim * 2, 4, 2, 1),
            self._make_crit_block(d_dim * 2, d_dim * 4, 4, 2, 1),
            self._make_crit_block(d_dim * 4, d_dim * 8, 4, 2, 1),
            self._make_crit_block(d_dim * 8, d_dim * 16, 4, 2, 1),
            # Final layer
            nn.Conv2d(d_dim * 16, 1, 4, 1, 0)
        )

    def _make_crit_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        crit_pred = self.crit(image)
        # Flatten the output
        return crit_pred.view(len(crit_pred), -1)
