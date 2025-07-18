import torch

# --- Training Hyperparameters ---
N_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
BETA_1 = 0.5  # Adam optimizer beta1
BETA_2 = 0.9  # Adam optimizer beta2
CRITIC_CYCLES = 5  # Number of critic updates per generator update
GRADIENT_PENALTY_LAMBDA = 10

# --- Model Configuration ---
Z_DIM = 200  # Dimension of the latent noise vector
GENERATOR_DIM = 16
CRITIC_DIM = 16

# --- Data Configuration ---
IMAGE_SIZE = 128
DATASET_LIMIT = 10000  # Set to None to use the full dataset
NUM_WORKERS = 2  # Number of workers for DataLoader

# --- Environment Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Logging and Saving ---
WANDB_ENABLED = True  # Set to False to disable Weights & Biases logging
WANDB_PROJECT_NAME = "wgan-gp-celeba"
SHOW_STEP = 35  # How often to show generated images (in steps)
SAVE_STEP = 35  # How often to save model checkpoints (in steps)
CHECKPOINT_DIR = "checkpoints"
