import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image

class ScalogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['file']
        label = self.data.iloc[idx]['label']
        subfolder = 'AFIB' if label == 1 else 'nonAFIB'
        file_path = f"{self.root_dir}/{subfolder}/{file_name}"

        scalogram = np.load(file_path)
        scalogram_img = Image.fromarray((scalogram / np.max(scalogram) * 255).astype(np.uint8))

        scalogram_img = self.transform(scalogram_img)

        return scalogram_img, torch.tensor(label, dtype=torch.float32)
