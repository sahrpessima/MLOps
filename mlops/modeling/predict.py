from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import torch
import torch.nn.functional as F

from mlops.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    model, dataloader, device
):

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


if __name__ == "__main__":
    app()
