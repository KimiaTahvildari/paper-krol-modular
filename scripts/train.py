import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from datasets.scalogram_datasets import ScalogramDataset
from config import OUTPUT_DIR
from models.cnn import AFibBaselineCNN
from models.cnn import AFibOptimizedCNN

# Parameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset & DataLoader
dataset = ScalogramDataset(
    csv_file=f"{OUTPUT_DIR}/metadata.csv",
    root_dir=OUTPUT_DIR,
    transform=None  # We'll add transforms inside the Dataset if needed
)

# Split dataset: 90% train, 10% val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
model = AFibBaselineCNN().to(device)
#model = AFibOptimizedCNN().to(device)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1, 1)  # Ensure shape (B, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        preds = (outputs >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = accuracy_score(np.vstack(all_labels), np.vstack(all_preds))
    train_f1 = f1_score(np.vstack(all_labels), np.vstack(all_preds))
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {train_acc:.4f} - F1: {train_f1:.4f}")
    
    # Validation loop
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            preds = (outputs >= 0.5).float()
            val_preds.append(preds.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
    
    val_acc = accuracy_score(np.vstack(val_labels), np.vstack(val_preds))
    val_f1 = f1_score(np.vstack(val_labels), np.vstack(val_preds))
    print(f"Validation - Acc: {val_acc:.4f} - F1: {val_f1:.4f}")

# Save the model
torch.save(model.state_dict(), f"{OUTPUT_DIR}/cnn_model.pt")
print("✅ Model saved.")
