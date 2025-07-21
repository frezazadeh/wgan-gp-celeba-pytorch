import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from src.config import BATCH_SIZE, IMAGE_SIZE, DATASET_LIMIT, NUM_WORKERS

class Dataset(Dataset):
    def __init__(self, path, size=128, lim=10000):
        self.sizes = [size, size]
        # Collect image paths
        self.items = [os.path.join(path, f)
                      for f in os.listdir(path)[:lim]
                      if f.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Load and resize image
        img = Image.open(self.items[idx]).convert('RGB')  # PIL Image (H,W,C)
        img = T.Resize(self.sizes)(img)                  # PIL Image 128x128
        # Convert to numpy array and reorder axes to C×H×W
        arr = np.asarray(img, dtype=np.float32)          # H×W×C in float32
        arr = arr.transpose((2, 0, 1))                   # C×H×W
        # Scale to [0,1]
        tensor = torch.from_numpy(arr).div(255)
        tensor = tensor * 2.0 - 1.0
        return tensor


def get_celeba_dataloader() -> DataLoader:
    import zipfile, kagglehub
    print("[INFO] Downloading CelebA dataset via KaggleHub...")
    dataset_dir = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print(f"[INFO] Dataset downloaded to: {dataset_dir}")

    # Find image directory
    img_path = None
    for root, _, files in os.walk(dataset_dir):
        if any(f.lower().endswith('.jpg') for f in files):
            img_path = root
            break

    # Extract zipped images if needed
    zip_path = os.path.join(dataset_dir, 'img_align_celeba.zip')
    if img_path is None and os.path.isfile(zip_path):
        print(f"[INFO] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dataset_dir)
        for root, _, files in os.walk(dataset_dir):
            if any(f.lower().endswith('.jpg') for f in files):
                img_path = root
                break

    if img_path is None:
        raise FileNotFoundError(f"Could not find .jpg images in {dataset_dir}")

    print(f"[INFO] Using image directory: {img_path}")
    dataset = Dataset(path=img_path, size=IMAGE_SIZE, lim=DATASET_LIMIT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return loader
