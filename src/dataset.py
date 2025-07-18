import os
import zipfile
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.config import BATCH_SIZE, IMAGE_SIZE, DATASET_LIMIT, NUM_WORKERS

class CelebADataset(Dataset):
    """
    Custom PyTorch Dataset for the CelebA dataset.
    Handles loading images and applying transformations.
    """
    def __init__(self, path: str, size: int = 128, limit: int = None):
        self.path = path
        self.size = (size, size)
        
        all_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
        self.items = all_files[:limit] if limit is not None else all_files

        # Transformation pipeline to resize, convert to tensor, and normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.items[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

def get_celeba_dataloader() -> DataLoader:
    """
    Downloads, extracts, and prepares the CelebA dataset.
    Returns a PyTorch DataLoader for training.
    """
    print("[INFO] Downloading CelebA dataset via KaggleHub...")
    dataset_dir = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print(f"[INFO] Dataset downloaded to: {dataset_dir}")

    # Dynamically find the directory containing .jpg images
    img_path = None
    for root, _, files in os.walk(dataset_dir):
        if any(f.lower().endswith('.jpg') for f in files):
            img_path = root
            break  # Stop searching once images are found

    # Handle extraction if the image directory isn't found directly
    zip_path = os.path.join(dataset_dir, 'img_align_celeba.zip')
    if img_path is None and os.path.isfile(zip_path):
        print(f"[INFO] Extracting images from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dataset_dir)
        # Search again after extraction
        for root, _, files in os.walk(dataset_dir):
            if any(f.lower().endswith('.jpg') for f in files):
                img_path = root
                break

    # Raise an error if no images are found after all checks
    if img_path is None:
        raise FileNotFoundError(f"Could not find any .jpg images within the dataset directory: {dataset_dir}")

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
