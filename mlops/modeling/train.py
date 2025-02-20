import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def train_model(model, train_loader, test_loader, device, num_epochs=10, lr=1e-1, save_path='models/model.onnx'):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(lr=lr, params=model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2)
    
    logs = {'accuracy': [], 'loss': []}
    
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            model.train()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}')
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logs['accuracy'].append(accuracy)
        logs['loss'].append(loss.item())
        print(f'Validation Accuracy: {accuracy:.2f}%')
    
    # Экспорт модели в ONNX
    dummy_input = torch.randn(1, 3, 224, 285).to(device)
    torch.onnx.export(model, dummy_input, save_path, export_params=True, opset_version=11)
    print(f'Model exported to {save_path}')
    
    return logs
