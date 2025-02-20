import torch
from torchvision.transforms import v2

IMAGE_SIZE = (224, 285)

def get_transforms():
    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.RandomPerspective(distortion_scale=0.3, p=1.0),
        v2.RandomRotation(10),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=IMAGE_SIZE, antialias=True),
        v2.ToTensor(),
        lambda x: (x / 255)
    ])
    
    transforms_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Resize(size=IMAGE_SIZE, antialias=True),
        v2.ToTensor(),
        lambda x: (x / 255),
    ])
    
    return transforms_train, transforms_test
