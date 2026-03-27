import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import json

# step 1: -----------Load the data-------------

# preprocess image before feeding to model
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# load image from the folder
train_data = datasets.ImageFolder('data/Training', transform=transform)
test_data = datasets.ImageFolder('data/Testing', transform=transform)

train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
test_loader = DataLoader(test_data, batch_size=20)

print("Classes found: ", train_data.classes)

# step 2: --------------Load pretrained ResNet18----------------

# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers — don't retrain from scratch
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for 4 classes
model.fc = nn.Linear(model.fc.in_features, 4)

# move model to gpu if not available then cpu
model = model.to(device)

# step 3: ---------Train------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)

EPOCHS = 25

print("\nTraining started...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    # per batch training loop
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# step 4: ------------Test accuracy---------------

print("\nEvaluating on test set...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# step 5: ---------------save model and class names--------------

torch.save(model.state_dict(), 'brain_tumor_model.pth')

with open('classes.json', 'w') as f:
    json.dump(train_data.classes, f)

