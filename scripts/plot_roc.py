import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

from datasets.scalogram_datasets import ScalogramDataset
from models.cnn import AFibOptimizedCNN
from models.cnn import AFibBaselineCNN
from config import OUTPUT_DIR

# Parameters
BATCH_SIZE = 32

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
dataset = ScalogramDataset(
    csv_file=f"{OUTPUT_DIR}/metadata.csv",
    root_dir=OUTPUT_DIR,
    transform=None
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = AFibBaselineCNN().to(device)
#model = AFibOptimizedCNN().to(device)
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/cnn_model.pt", map_location=device))
model.eval()
print("✅ Model loaded.")

# Get true labels & predicted probabilities
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_probs.append(outputs.cpu().numpy())
        all_labels.append(labels.view(-1, 1).cpu().numpy())

y_true = np.vstack(all_labels)
y_probs = np.vstack(all_probs)

# ROC curve & AUC
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Optional: Save the plot
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
