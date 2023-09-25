import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from customDataset import PlantDiseaseDataset
from model import CustomResNet, evaluate_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 32

train_dataset = PlantDiseaseDataset(root_dir='../new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = PlantDiseaseDataset(root_dir='../new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
model = CustomResNet(num_classes=38).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    train_accuracy = (correct_predictions / total_samples) * 100
    train_accuracies.append(train_accuracy)

    val_loss, val_accuracy = evaluate_model(model, valid_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_loss:.4f} - Training Accuracy: {train_accuracy:.2f}% - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%')

torch.save(model.state_dict(), '../models/custom_resnet_model.pth')
