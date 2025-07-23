
#!/usr/bin/env python3

import os
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

# ─── Config and Hyperparameters ─────────────────────────────
DATA_ROOT     = '/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/Dataset-vs-CNN'
CLASS_FOLDERS = {
    'Glioma-Backup':    0,
    'Meningioma-Backup':1,
    'Pituitary-Backup':  2,
    'No-Tumor':          3,
    'ALL-Tumor':        4
}
# Indices of the two classes to compare
COMPARE_A      = 2  # e.g., 0 for Glioma-Backup
COMPARE_B      = 3 # e.g., 1 for Meningioma-Backup

IMG_SIZE       = 8         # resized to IMG_SIZE x IMG_SIZE
N_QUBITS       = 5         # quantum features/qubits
N_LAYERS       = 2         # number of convolution/pooling blocks
N_CLASSES      = 2         # binary classification output size
BATCH_SIZE     = 32
EPOCHS         = 700
LR             = 0.0015
OUTPUT_ROOT    = './outputs'
RANDOM_STATE   = 123

# ─── Utility Functions ─────────────────────────────────────

def load_topdown_images(data_root, class_folders, img_size):
    X, y = [], []
    for class_name, class_idx in class_folders.items():
        img_dir = os.path.join(data_root, class_name, '1')
        if not os.path.isdir(img_dir):
            print(f"Warning: Could not find folder {img_dir}")
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            fpath = os.path.join(img_dir, fname)
            try:
                img = Image.open(fpath).convert('L')
                img = img.resize((img_size, img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten()); y.append(class_idx)
            except Exception as e:
                print(f'Could not process {fpath}: {e}')
    return np.array(X), np.array(y)

# ─── QCNN Blocks ───────────────────────────────────────────
def conv_block(weights, wires):
    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RZ(weights[i], wires=wires[(i+1)%len(wires)])
        qml.CNOT(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RX(np.pi/2, wires=wires[i]);   qml.RX(np.pi/2, wires=wires[(i+1)%len(wires)])
        qml.CNOT(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RX(-np.pi/2, wires=wires[i]);  qml.RX(-np.pi/2, wires=wires[(i+1)%len(wires)])
        qml.CZ(wires=[wires[i], wires[(i+1)%len(wires)]])
        qml.RZ(weights[i+2], wires=wires[(i+1)%len(wires)])
        qml.CZ(wires=[wires[i], wires[(i+1)%len(wires)]])

def pool_block(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CRY(weights[0], wires=[wires[1], wires[0]])
    qml.CRX(weights[1], wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[0]])

# ─── QCNN Circuit ──────────────────────────────────────────
def qcnn_circuit(x, conv, pool, n_layers=N_LAYERS, n_qubits=N_QUBITS):
    wires = list(range(n_qubits))
    qml.AngleEmbedding(x, wires=wires, rotation='Y')
    for l in range(n_layers):
        conv_block(conv[l], wires)
        pool_block(pool[l], wires)
        wires = wires[1:]
    return [qml.expval(qml.PauliZ(w)) for w in wires[:N_CLASSES]]

# ─── PennyLane Setup ───────────────────────────────────────
dev = qml.device('default.qubit', wires=N_QUBITS)
weight_shapes = {'conv': (N_LAYERS, 3*N_QUBITS), 'pool': (N_LAYERS, 2)}

@qml.qnode(dev, interface='torch')
def qnode(inputs, conv, pool):
    return qcnn_circuit(inputs, conv, pool)

qcnn_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ─── Hybrid QCNN Model ─────────────────────────────────────
class HybridQCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.qc = qcnn_layer
        self.fc = nn.Linear(n_classes, n_classes)
    def forward(self, x):
        return self.fc(self.qc(x))

# ─── Loss & Accuracy ───────────────────────────────────────
def cce_loss(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def accuracy(outputs, targets):
    return (outputs.argmax(dim=1) == targets).float().mean().item()

# ─── Main: Single Pair Comparison ──────────────────────────
if __name__ == '__main__':
    np.random.seed(RANDOM_STATE); torch.manual_seed(RANDOM_STATE)
    X, y = load_topdown_images(DATA_ROOT, CLASS_FOLDERS, IMG_SIZE)
    idx_to_name = {v: k for k, v in CLASS_FOLDERS.items()}

    # Use global COMPARE_A and COMPARE_B
    a, b = COMPARE_A, COMPARE_B
    name_a = idx_to_name[a].replace(' ', '_')
    name_b = idx_to_name[b].replace(' ', '_')
    print(f"\n=== Training {name_a} vs {name_b} ===")

    # Filter and remap labels
    mask = np.isin(y, [a, b])
    X_sel, y_sel = X[mask], y[mask]
    y_bin = np.where(y_sel == a, 0, 1)

    # PCA + normalize to [0, π]
    pca = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE)
    X_red = pca.fit_transform(X_sel)
    mn, mx = X_red.min(axis=0), X_red.max(axis=0)
    X_angles = ((X_red - mn)/(mx-mn+1e-10)) * np.pi

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_angles, y_bin, test_size=0.2, stratify=y_bin, random_state=RANDOM_STATE)

    # Balance
    unique, counts = np.unique(y_tr, return_counts=True)
    min_c = counts.min()
    idxs = np.hstack([np.random.choice(np.where(y_tr == c)[0], min_c, replace=False)
                      for c in unique])
    np.random.shuffle(idxs)
    X_tr, y_tr = X_tr[idxs], y_tr[idxs]

    # DataLoaders
    tr_ds = TensorDataset(torch.from_numpy(X_tr.astype(np.float32)), torch.from_numpy(y_tr))
    te_ds = TensorDataset(torch.from_numpy(X_te.astype(np.float32)), torch.from_numpy(y_te))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE)

    # Model & optimizer
    model = HybridQCNN().float()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Training
    losses, accs = [], []
    for ep in range(1, EPOCHS + 1):
        model.train()
        Ls, As = [], []
        for xb, yb in tr_ld:
            opt.zero_grad()
            out = model(xb)
            loss = cce_loss(out, yb)
            loss.backward()
            opt.step()
            Ls.append(loss.item()); As.append(accuracy(out, yb))
        L, A = np.mean(Ls), np.mean(As)
        losses.append(L); accs.append(A)
        print(f"Epoch {ep}/{EPOCHS} | Loss={L:.4f} | Acc={A:.3f}")

    # Evaluation
    model.eval()
    Ls, As = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            out = model(xb)
            Ls.append(cce_loss(out, yb).item()); As.append(accuracy(out, yb))
    test_loss, test_acc = np.mean(Ls), np.mean(As)
    print(f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.3f}")

    # Save
    base = f"{name_a}_vs_{name_b}"
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    odir = os.path.join(OUTPUT_ROOT, base + '_' + now)
    os.makedirs(odir, exist_ok=True)

    # Plot
    plt.figure()
    plt.plot(losses, label='loss'); plt.plot(accs, label='acc'); plt.legend()
    plt.savefig(os.path.join(odir, base + '_curve.png'))
