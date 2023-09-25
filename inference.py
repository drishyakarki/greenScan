from torchvision import transforms
import torch
from src.model import CustomResNet
from src.customDataset import PlantDiseaseDataset
from torchvision.datasets import ImageFolder
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

train_dataset = PlantDiseaseDataset(root_dir='data/New Plant Diseases Dataset(Augmented)/train', transform=transform)

classes = train_dataset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomResNet(num_classes=38).to(device)

model.load_state_dict(torch.load("models/custom_resnet_model.pth"))

model.eval()

def predict_image(img, model):
    xb = img.unsqueeze(0).to(device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return classes[preds[0].item()]

test_dir = 'test'
test = ImageFolder(test_dir, transform=transform)
test_images = sorted(os.listdir(test_dir + '/test'))

for i, (img, label) in enumerate(test):
    print('Label:', test_images[i], ', Predicted:', predict_image(img, model))

