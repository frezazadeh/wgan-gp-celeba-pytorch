# WGAN-GP for High-Resolution Face Generation 👱‍♀️👨‍🦰

This project is a PyTorch implementation of a Wasserstein GAN with Gradient Penalty (WGAN-GP) designed to generate high-resolution celebrity faces from the CelebA dataset. The architecture is based on Deep Convolutional GANs (DCGANs).

This repository refactors the original [Google Colab notebook by Javier Ideami](https://colab.research.google.com/drive/11rmZGb1qvyddLSmqBuLMn8fC87nmmwSc) into a structured, professional, and easily maintainable Python project.

![Latent Space Morphing](https://i.imgur.com/uQAmIS4.gif)
*(Example of latent space interpolation)*

---

## ✨ Features

-   **WGAN-GP:** Utilizes the Wasserstein loss with Gradient Penalty for stable training and high-quality image generation.
-   **DCGAN Architecture:** Employs a deep convolutional structure for both the Generator and the Critic.
-   **KaggleHub Integration:** Automatically downloads and prepares the CelebA dataset using the `kagglehub` library.
-   **Weights & Biases Logging:** Integrated experiment tracking for losses, generated images, and model graphs.
-   **Modular Codebase:** The code is split into logical modules for configuration, data handling, models, and training logic.
-   **Image Generation & Morphing:** Includes a script to generate new images and create smooth latent space interpolations from a trained model.

---

## 📂 Project Structure

The project is organized to separate concerns, making it easy to navigate and modify.

```
wgan-gp-celeba/
├── checkpoints/         # Saved model checkpoints
├── src/
│   ├── config.py        # All hyperparameters
│   ├── dataset.py       # Data loading (KaggleHub)
│   ├── models.py        # Generator & Critic classes
│   ├── trainer.py       # The main WGAN-GP training loop
│   └── utils.py         # Helper functions
├── generate.py          # Script for image generation
├── main.py              # Main script to start training
├── requirements.txt     # Dependencies
└── README.md            # You are here
```

---

## 🚀 Getting Started

### 1. Prerequisites

-   Python 3.8+
-   `pip` and `virtualenv` (recommended)
-   A Kaggle account and API token (`kaggle.json`)
-   A Weights & Biases account (optional, for logging)

### 2. Installation

Clone the repository and set up the environment.

```bash
# Clone the project
git clone [https://github.com/YOUR_USERNAME/wgan-gp-celeba.git](https://github.com/YOUR_USERNAME/wgan-gp-celeba.git)
cd wgan-gp-celeba

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required libraries
pip install -r requirements.txt
```

### 3. API Keys Setup

**Kaggle:**
To download the dataset, you need your Kaggle API key.
1.  Go to your Kaggle account settings and click "Create New Token" to download `kaggle.json`.
2.  Place the `kaggle.json` file in the required location:
    -   **Linux/macOS:** `~/.kaggle/kaggle.json`
    -   **Windows:** `C:\Users\<Your-Username>\.kaggle\kaggle.json`

**Weights & Biases (Optional):**
If you want to use W&B for logging (`WANDB_ENABLED = True` in `src/config.py`), you'll need to log in. The first time you run the training script, you will be prompted to enter your API key.

```bash
# You can also log in manually beforehand
wandb login
```

---

## 💡 How to Use

### Training the Model

To start training the WGAN-GP from scratch, simply run `main.py`. The script will automatically download the dataset, initialize the models, and begin training.

```bash
python main.py
```

-   Progress will be displayed in the console using `tqdm`.
-   If W&B is enabled, a link to your experiment dashboard will be printed.
-   Model checkpoints will be saved periodically to the `checkpoints/` directory.

### Generating Images from a Checkpoint

Once you have a trained model, use the `generate.py` script to create new images or a latent space morphing animation. You must provide the path to a generator checkpoint file.

```bash
# Example: Generate images and a morphing grid from a saved step
python generate.py --checkpoint checkpoints/G_step_105.pth
```
This will display a grid of newly generated faces and a grid showing a smooth transition between different points in the latent space.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

-   This project is a refactored and enhanced version of the GAN course material by **Javier Ideami**.
-   The implementation is based on the paper [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Gulrajani et al.
