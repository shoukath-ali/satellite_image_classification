import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data_loader import SatelliteDataset
from model import TerrainClassifier
from torch.utils.data import DataLoader, random_split

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

    dataset = SatelliteDataset(data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, num_classes=4):
    """
    Train the terrain classification model.
    Args:
        data_dir (str): Path to the dataset directory.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for DataLoader.
        learning_rate (float): Learning rate for the optimizer.
        num_classes (int): Number of terrain classes.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)

    model = TerrainClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/terrain_classifier.pth")
    print("Model saved to models/terrain_classifier.pth")

if __name__ == "__main__":

    data_dir = "data"
    train_model(data_dir)
