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
        """
        Initializes the dataset. The transform pipeline is removed from here
        and implemented manually in __getitem__.
        """
        self.path = path
        self.sizes = (size, size)
        
        all_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
        self.items = all_files[:limit] if limit else all_files

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        Loads an image, processes it manually, and scales it to [0, 1].
        This method reverts to the notebook's original normalization.
        """
        # Load image and resize
        img = Image.open(self.items[idx]).convert('RGB')
        img_resized = transforms.Resize(self.sizes)(img)
        
        # Convert to NumPy array and transpose dimensions from (H, W, C) to (C, H, W)
        arr = np.transpose(np.asarray(img_resized), (2, 0, 1)).astype(np.float32)
        
        # Convert to a PyTorch tensor and scale pixel values to the [0, 1] range
        tensor = torch.from_numpy(arr).div(255)
        
        return tensor

def get_celeba_dataloader() -> DataLoader:
    """
    Downloads, extracts, and prepares the CelebA dataset.
    Returns a PyTorch DataLoader.
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

    # Handle extraction if the directory isn't found directly
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
