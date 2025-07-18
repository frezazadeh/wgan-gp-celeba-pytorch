import os
import zipfile
import kagglehub
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

from src.config import BATCH_SIZE, IMAGE_SIZE, DATASET_LIMIT, NUM_WORKERS

class CelebADataset(Dataset):
    def __init__(self, path: str, size: int = 128, limit: int = None):
        self.path = path
        self.size = (size, size)
        
        all_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
        self.items = all_files[:limit] if limit else all_files

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(), # This converts PIL image to tensor and scales to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
        ])
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

def get_celeba_dataloader() -> DataLoader:
    """
    Downloads, extracts, and prepares the CelebA dataset.
    Returns a PyTorch DataLoader.
    """
    print("[INFO] Downloading CelebA dataset via KaggleHub...")
    dataset_dir = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print(f"[INFO] Dataset downloaded to: {dataset_dir}")

    # Check for zip and extract if needed
    zip_path = os.path.join(dataset_dir, 'img_align_celeba.zip')
    img_dir_name = 'img_align_celeba'
    img_path = os.path.join(dataset_dir, img_dir_name)

    if not os.path.isdir(img_path):
        if os.path.isfile(zip_path):
            print(f"[INFO] Extracting images from {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(dataset_dir)
            print(f"[INFO] Extracted images to: {img_path}")
        else:
            raise FileNotFoundError(f"Neither image directory '{img_dir_name}' nor zip file '{zip_path}' found.")
    
    print(f"[INFO] Using image directory: {img_path}")
    
    # Create dataset and dataloader
    dataset = CelebADataset(path=img_path, size=IMAGE_SIZE, limit=DATASET_LIMIT)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    return dataloader
