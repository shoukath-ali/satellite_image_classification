import torch
import os
import numpy as np
from torchvision import transforms, models
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from model import get_model

def predict_terrain_deep_learning(model_name, model_path, image_path, num_classes=4):
    """
    Predict terrain type using a deep learning model.
    Args:
        model_name (str): Name of the model ('resnet' or 'custom').
        model_path (str): Path to the trained model.
        image_path (str): Path to the input image.
        num_classes (int): Number of output classes.
    Returns:
        str: Predicted terrain class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {model_name} model for prediction...")
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_names = ["cloudy", "desert", "green_area", "water"]

    return class_names[predicted.item()]


def predict_terrain_traditional_ml(models, image_path, feature_extractor_model, scaler):
    """
    Predict terrain type using traditional ML models.
    Args:
        models (dict): Trained traditional ML models.
        image_path (str): Path to the input image.
        feature_extractor_model (torch.nn.Module): Pre-trained ResNet model as feature extractor.
        scaler (StandardScaler): Fitted scaler for feature standardization.
    Returns:
        dict: Predictions from all traditional ML models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    feature_extractor_model.to(device)
    feature_extractor_model.eval()
    with torch.no_grad():
        features = feature_extractor_model(image)
        features = features.view(features.size(0), -1).cpu().numpy()

    features = scaler.transform(features)

    predictions = {}
    for name, model in models.items():
        predicted_class = model.predict(features)
        predictions[name] = predicted_class[0]

    class_names = ["cloudy", "desert", "green_area", "water"]

    return {name: class_names[pred] for name, pred in predictions.items()}



if __name__ == "__main__":
    image_path = "data/test.jpeg"  

    print("Predicting using ResNet...")
    resnet_prediction = predict_terrain_deep_learning(
        model_name="resnet",
        model_path="models/resnet_terrain_classifier.pth",
        image_path=image_path
    )
    print(f"ResNet Prediction: {resnet_prediction}")

    print("Predicting using Custom CNN...")
    custom_cnn_prediction = predict_terrain_deep_learning(
        model_name="custom",
        model_path="models/custom_terrain_classifier.pth",
        image_path=image_path
    )
    print(f"Custom CNN Prediction: {custom_cnn_prediction}")

    print("Extracting features using ResNet for Traditional ML Models...")
    feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])

    from data_loader import extract_features
    features, labels = extract_features("data")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    traditional_ml_models = {
        "KNN": KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train),
        "SVM": SVC(kernel="rbf", probability=True).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    }

    print("Predicting using Traditional ML Models...")
    traditional_ml_predictions = predict_terrain_traditional_ml(
        models=traditional_ml_models,
        image_path=image_path,
        feature_extractor_model=feature_extractor,
        scaler=scaler
    )

    for model_name, prediction in traditional_ml_predictions.items():
        print(f"{model_name} Prediction: {prediction}")
