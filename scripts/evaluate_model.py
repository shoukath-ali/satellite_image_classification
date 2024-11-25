import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import SatelliteDataset
from model import get_model

def evaluate_model(data_dir, model_name, model_path, batch_size=32, num_classes=4):
    """
    Evaluate the specified model.
    Args:
        data_dir (str): Path to dataset directory (test data).
        model_name (str): Name of the model architecture.
        model_path (str): Path to the trained model.
        batch_size (int): Batch size for DataLoader.
        num_classes (int): Number of terrain classes.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    dataset = SatelliteDataset(data_dir)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Classification Report
    print(f"Results for {model_name}:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    data_dir = "data"
    models_to_evaluate = [
        {"name": "resnet", "path": "models/resnet_terrain_classifier.pth"},
        {"name": "custom", "path": "models/custom_terrain_classifier.pth"}
    ]

    for model_info in models_to_evaluate:
        evaluate_model(data_dir, model_info["name"], model_info["path"])
