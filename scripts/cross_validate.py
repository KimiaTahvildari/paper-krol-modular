import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

from datasets.scalogram_datasets import ScalogramDataset
from models.cnn import AFibBaselineCNN
from models.cnn import AFibOptimizedCNN


from config import OUTPUT_DIR

# Parameters
BATCH_SIZE = 32
EPOCHS = 10    # Cross-validation usually uses fewer epochs per fold for speed
LEARNING_RATE = 0.001
NUM_FOLDS = 5

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset
dataset = ScalogramDataset(
    csv_file=f"{OUTPUT_DIR}/metadata.csv",
    root_dir=OUTPUT_DIR,
    transform=None
)

# Cross-validation setup
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# Metrics storage
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n🔄 Fold {fold+1}/{NUM_FOLDS}")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Model a-base line b-optimized vesion 
    model = AFibBaselineCNN().to(device)
   # model = AFibOptimizedCNN().to(device) # Uncomment to use the optimized model


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    # Validation
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

    y_true = np.vstack(val_labels)
    y_pred = np.vstack(val_preds)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"Fold {fold+1} Results - Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    fold_metrics.append({'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall})

# Final results
accs = [m['acc'] for m in fold_metrics]
f1s = [m['f1'] for m in fold_metrics]
precisions = [m['precision'] for m in fold_metrics]
recalls = [m['recall'] for m in fold_metrics]

print("\n✅ Cross-Validation Completed:")
print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1-score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
