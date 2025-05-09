import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

from datasets.scalogram_datasets import ScalogramDataset
from models.cnn import AFibOptimizedCNN
from models.cnn import AFibBaselineCNN
from config import OUTPUT_DIR

# Parameters
BATCH_SIZE = 32

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
dataset = ScalogramDataset(
    csv_file=f"{OUTPUT_DIR}/metadata.csv",
    root_dir=OUTPUT_DIR,
    transform=None
)

# Full dataset loader (you can also use a test split if available)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = AFibBaselineCNN().to(device)
#model = AFibOptimizedCNN().to(device)

model.load_state_dict(torch.load(f"{OUTPUT_DIR}/cnn_model.pt", map_location=device))
model.eval()
print("✅ Model loaded.")

# Collect predictions & labels
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1, 1)
        outputs = model(inputs)
        preds = (outputs >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

y_true = np.vstack(all_labels)
y_pred = np.vstack(all_preds)

# Compute metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"\n✅ Evaluation Results:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['nonAFIB', 'AFIB'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
