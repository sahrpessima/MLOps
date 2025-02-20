import torch
import torch.nn.functional as F

def predict(model, dataloader, device):
    """Функция для инференса модели."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)
            predictions.extend(predicted_classes.cpu().numpy())
    return predictions