import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import json

EPOCHS = 25

# step 1: -----------Load the data-------------

# preprocess image before feeding to model
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# Augmentation only for training split
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_train_data = datasets.ImageFolder('data/Training', transform=train_transform)
test_data = datasets.ImageFolder('data/Testing', transform=transform)

# split training set: 75% training and 25% validation
val_size   = int(0.25 * len(full_train_data))
train_size = len(full_train_data) - val_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

# Give the val split clean (no-augmentation) transforms
val_data.dataset = datasets.ImageFolder('data/Training', transform=transform)

train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=20)
test_loader  = DataLoader(test_data,  batch_size=20)

print("Classes:", full_train_data.classes)
print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_data)}")

# step 2: --------------Load pretrained ResNet18----------------

# device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all layers — don't retrain from scratch
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer 3 as well
for param in model.layer3.parameters():
    param.requires_grad = True

# Unfreeze last residual block (layer 4) for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer for 4 classes
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 4)
)

# move model to gpu if not available then cpu
model = model.to(device)

# step 3: ---------Train with validation tracking------------

# loss function using Cross Entropy Loss 
criterion = nn.CrossEntropyLoss()

# optimizer using Adam optimizer
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 3e-5},
    {'params': model.layer4.parameters(), 'lr': 5e-5},
    {'params': model.fc.parameters(),     'lr': 3e-4},
], weight_decay=1e-4)

# Reduce LR when val loss stops improving
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)


history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0
early_stop_patience = 7
epoch_no_improve = 0

print("\nTraining started...")
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
print("-" * 65)

for epoch in range(EPOCHS):
    # training started
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    # per batch training loop
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = 100 * train_correct / train_total

    # validation phase
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss    += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total   += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc  = 100 * val_correct / val_total

    # ── Log to history ────────────────────────
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    # ── Step scheduler on val loss ────────────
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(f"{epoch+1:>5} | {avg_train_loss:>10.4f} | {avg_train_acc:>8.2f}% | "
          f"{avg_val_loss:>8.4f} | {avg_val_acc:>6.2f}% | {current_lr:>8.6f}")

    # ── Save best model ───────────────────────
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), 'brain_tumor_model_best.pth')
        print(f"         ✔ New best val accuracy: {best_val_acc:.2f}% — model saved")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # ── Early stopping ────────────────────────
    if epochs_no_improve >= early_stop_patience:
        print(f"\nEarly stopping triggered — no improvement for {early_stop_patience} epochs.")
        break

# step 4: ------------Test accuracy---------------

print("\nLoading best model for final test evaluation...")
model.load_state_dict(torch.load('brain_tumor_model_best.pth'))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
print(f"Best Val Accuracy seen during training: {best_val_acc:.2f}%")

# step 5: ---------------save model and class names--------------

torch.save(model.state_dict(), 'brain_tumor_model.pth')

with open('classes.json', 'w') as f:
    json.dump(full_train_data.classes, f)

with open('training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\nSaved: brain_tumor_model.pth | classes.json | training_history.json")

