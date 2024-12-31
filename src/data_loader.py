import os
import torch
from glob import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import collections
collections.Iterable = collections.abc.Iterable


def add_gaussian_noise(x):
    return x + 0.1 * torch.randn_like(x)

def normalize_tensor(x):
    return torch.clamp(x, 0., 1.)

class NonClassDataLoader(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_paths = glob(os.path.join(root_dir, '*.jpg')) + \
                           glob(os.path.join(root_dir, '*.jpeg')) + \
                           glob(os.path.join(root_dir, '*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image

def get_data_loaders(data_path, batch_size):
    # Define transforms
    transform_noised = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(add_gaussian_noise),
        transforms.Lambda(normalize_tensor),
    ])

    transform_clean = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create datasets
    noised_dataset = NonClassDataLoader(root_dir=data_path, transforms=transform_noised)
    clean_dataset = NonClassDataLoader(root_dir=data_path, transforms=transform_clean)

    # Create dataloaders
    train_data = DataLoader(
        dataset=noised_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    clean_data = DataLoader(
        dataset=clean_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    return train_data, clean_data
