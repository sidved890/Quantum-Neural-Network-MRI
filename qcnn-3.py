#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
import pennylane as qml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ─── Config and Hyperparameters ──────────────────────────────
DATA_ROOT = '/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/Dataset-vs-CNN'
CLASS_FOLDERS = {
    "Glioma-Backup": 0,
    "Meningioma-Backup": 1,
    "Pituitary-Backup": 2
}
IMG_SIZE = 8         # <= Input images will be resized to square IMG_SIZE x IMG_SIZE
N_QUBITS = 6         # Number of quantum features/qubits for circuit (suggest: same as PCA n_components)
N_CLASSES = 3
N_LAYERS = 2        # Number of quantum Convolution/Pooling blocks
BATCH_SIZE = 16
EPOCHS = 700
LR = 0.00175
OUTPUT_ROOT = "./outputs"
RANDOM_STATE = 123

# ─── 1. Data Loader ──────────────────────────────────────────
def load_topdown_images(data_root, class_folders, img_size):
    X, y = [], []
    for class_name, class_idx in class_folders.items():
        img_dir = os.path.join(data_root, class_name, "1")
        if not os.path.isdir(img_dir):
            print(f"Warning: Could not find folder {img_dir}")
            continue
        for fname in os.listdir(img_dir):

            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            fpath = os.path.join(img_dir, fname)
            try:
                img = Image.open(fpath).convert('L')
                img = img.resize((img_size, img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(class_idx)
            except Exception as e:
                print(f"Could not process {fpath}: {e}")
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images, shape: {X.shape}")
    return X, y

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

X, y = load_topdown_images(DATA_ROOT, CLASS_FOLDERS, IMG_SIZE)

# ─── 2. PCA and Angle Normalization ─────────────────────────
pca = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE)
X_reduced = pca.fit_transform(X)
print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")

def normalize_to_pi(X):
    minv = X.min(axis=0, keepdims=True)
    maxv = X.max(axis=0, keepdims=True)
    normed = (X - minv) / (maxv - minv + 1e-10)
    return normed * np.pi

X_angles = normalize_to_pi(X_reduced)

# ─── 3. Train/Test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_angles, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Train samples:", len(y_train), "Test samples:", len(y_test))
print("Class balance (train):", np.bincount(y_train))
print("Class balance (test) :", np.bincount(y_test))

# ─── 4. Build Torch DataLoaders ──────────────────────────────
train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train))
test_ds  = TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ─── 5. PennyLane: Modular QCNN Circuit ─────────────────────
def conv_block(weights, wires):
    for i in range(len(wires)):
        # qml.RY(weights[i], wires=wires[i+1])
        # qml.RY(weights[i+1], wires=wires[i])
        # qml.CNOT(wires=[wires[i], wires[i+1]])
        qml.CNOT(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RZ(weights[i], wires=wires[(i+1)%len(wires)])
        qml.CNOT(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RX(np.pi/2, wires=wires[i])
        qml.RX(np.pi/2, wires=wires[(i+1)%len(wires)])
        qml.CNOT(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RX(-np.pi/2, wires=wires[i])
        qml.RX(-np.pi/2, wires=wires[(i+1)%len(wires)])
        qml.CZ(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RZ(weights[i+2], wires=wires[(i+1)%len(wires)])
        qml.CZ(wires=[wires[i], wires[(i+1)%len(wires)]])





def pool_block(weights, wires):
    # weights: shape (4,) -- can be adjusted as needed

    # Local rotations on both wires
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])

    # Entangle, controlled rotations, and additional CNOT
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CRY(weights[0], wires=[wires[1], wires[0]])
    qml.CRX(weights[1], wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[0]])

    

def qcnn_circuit(x, conv, pool, n_layers=N_LAYERS, n_qubits=N_QUBITS):
    wires = list(range(n_qubits))
    qml.AngleEmbedding(x, wires=wires, rotation='Y')
    for l in range(n_layers):
        conv_block(conv[l], wires)
        pool_block(pool[l], wires)
        wires = wires[1:]   
    return [qml.expval(qml.PauliZ(w)) for w in wires[:N_CLASSES]]

dev = qml.device("default.qubit", wires=N_QUBITS)
weight_shapes = {
    "conv": (N_LAYERS, 3*N_QUBITS),
    "pool": (N_LAYERS, 2)
}

@qml.qnode(dev, interface="torch")
def qnode(inputs, conv, pool):
    return qcnn_circuit(inputs, conv, pool)

qcnn_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ─── 6. PyTorch Hybrid Model ─────────────────────────────────
class HybridQCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.qc = qcnn_layer
        self.fc = nn.Linear(n_classes, n_classes)
    def forward(self, x):
        x = self.qc(x)
        return self.fc(x)

# ─── 7. Loss, accuracy, result logging ───────────────────────
def cce_loss(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item()

# ─── 8. Training and Evaluation Loops ────────────────────────
model = HybridQCNN().float()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_losses, train_accs = [], []

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H-%M-%S")
NEW_ROOT = os.path.join(OUTPUT_ROOT, date_str)
os.makedirs(NEW_ROOT, exist_ok=True)

print("==== Starting Training ====")
for epoch in range(1, EPOCHS+1):
    model.train()
    batch_losses, batch_accs = [], []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb.float())
        loss = cce_loss(out, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        batch_accs.append(accuracy(out, yb))
    L = np.mean(batch_losses)
    A = np.mean(batch_accs)
    train_losses.append(L)
    train_accs.append(A)
    print(f"Epoch {epoch:3d} | Loss={L:.4f} | Acc={A:.3f}")

# ─── 9. Save Training Curve ──────────────────────────────────
plot_path = os.path.join(NEW_ROOT, f"QCNN-{time_str}-graph.png")
plt.figure(figsize=(6,3))
plt.subplot(1,2,1); plt.plot(train_losses); plt.title("Train Loss")
plt.subplot(1,2,2); plt.plot(train_accs); plt.title("Train Accuracy")
plt.tight_layout(); plt.savefig(plot_path)
plt.close()
print(f"Saved training plots to {plot_path}")

# ─── 10. Final Evaluation ────────────────────────────────────
def test_eval(model, test_loader):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.float())
            losses.append(cce_loss(out, yb).item())
            accs.append(accuracy(out, yb))
    return np.mean(losses), np.mean(accs)

test_loss, test_acc = test_eval(model, test_loader)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Acc:  {test_acc:.4f}")

# ─── 11. Save Metrics and Best Epoch ─────────────────────────
file_path = os.path.join(NEW_ROOT, f"QCNN-{time_str}.txt")
with open(file_path, "w") as f:
    f.write("Test type: 3-class classification with hybrid QCNN\n")
    f.write("Hyperparameters:\n")
    f.write(f"IMG_SIZE      = {IMG_SIZE}\n")
    f.write(f"N_QUBITS      = {N_QUBITS}\n")
    f.write(f"BATCH_SIZE    = {BATCH_SIZE}\n")
    f.write(f"EPOCHS        = {EPOCHS}\n")
    f.write(f"LR            = {LR}\n\n")
    f.write("Detailed test predictions:\n")
    f.write(f"Final Test Loss: {test_loss:.4f}\n")
    f.write(f"Final Test Acc: {test_acc:.4f}\n")
    best_epoch = np.argmax(train_accs)+1
    f.write(f"Maximum training Acc: {np.max(train_accs):.4f} at epoch {best_epoch}\n")
print(f"Saved test outputs to {file_path}")
