import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SatelliteDataset(Dataset):
    def __init__(self, data_dir, img_size=(224, 224), transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            img_size (tuple): Resize images to this size.
            transform: Torchvision transform for data augmentation and preprocessing.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(data_dir, class_name)
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    
def get_data_loaders(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Prepare PyTorch DataLoaders for training and testing.
    Args:
        data_dir (str): Path to dataset directory.
        batch_size (int): Batch size for DataLoader.
        img_size (tuple): Image resizing dimensions.
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects.
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SatelliteDataset(data_dir, img_size=img_size, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    data_dir = "data/satellite_images"
    train_loader, test_loader = get_data_loaders(data_dir)

    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")

