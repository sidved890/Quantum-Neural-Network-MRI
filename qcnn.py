#!/usr/bin/env python3
import os, time
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()
(date_str := now.strftime("%Y-%m-%d"))
(time_str := now.strftime("%H-%M-%S"))
# ─── Hyperparameters ─────────────────────────────────────────────────────────
DATA_ROOT     = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/Dataset-vs-CNN"
OUTPUT_ROOT  = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/outputs"
CLASS_FOLDERS = {"Glioma-Backup":0, "Meningioma-Backup":1, "Pituitary-Backup":2}
IMG_SIZE      = 8      # 16×16 = 256 pixels
PCA_COMP      = 16     # 64 features -> 6 qubits (2^6=64)
BATCH_SIZE    = 32
EPOCHS        = 200
LR            = 0.00175
RANDOM_STATE  = 180 

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run QCNN with custom hyperparams")
    parser.add_argument("--img_size",   type=int,   default=8,     help="Image dimension (square)")
    parser.add_argument("--pca_comp",   type=int,   default=16,    help="Number of PCA components")
    parser.add_argument("--batch_size", type=int,   default=32,    help="Training batch size")
    parser.add_argument("--epochs",     type=int,   default=100,   help="Number of epochs")
    parser.add_argument("--lr",         type=float, default=0.003, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()

    IMG_SIZE   = args.img_size
    PCA_COMP   = args.pca_comp
    BATCH_SIZE = args.batch_size
    EPOCHS     = args.epochs
    LR         = args.lr

    # … rest of your training/testing code …
    # make sure you reference these variables instead of hard-coded ones

if __name__ == "__main__":
    main()

def load_topdown(root, folders):
    X, y = [], []
    for cls, lbl in folders.items():
        view_dir = os.path.join(root, cls, "1")
        if not os.path.isdir(view_dir):
            print(f"Missing view folder: {view_dir}")
            continue
        for fn in os.listdir(view_dir):
            if fn.lower().endswith((".jpg",".png",".jpeg")):
                img = Image.open(os.path.join(view_dir, fn))\
                         .convert("L")\
                         .resize((IMG_SIZE, IMG_SIZE))
                X.append(np.array(img).flatten()/255.0)
                y.append(lbl)
    X = np.array(X); y = np.array(y)
    print(f"Loaded {len(X)} top-down images.")
    return X, y

print("Loading data…")
X, y = load_topdown(DATA_ROOT, CLASS_FOLDERS)

# ─── 2) Binary filter & train/test split ─────────────────────────────────────
mask = np.isin(y, [0,1])  # Glioma vs Meningioma
X, y = X[mask], y[mask]
y = (y == 1).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")

pca = PCA(n_components=PCA_COMP)
X_tr_pca = pca.fit_transform(X_tr)
X_te_pca = pca.transform(X_te)
print(f"PCA explained variance sum: {pca.explained_variance_ratio_.sum():.3f}")

def normalize_angles(mat):
    mins = mat.min(axis=1, keepdims=True)
    maxs = mat.max(axis=1, keepdims=True)
    return np.pi * (mat - mins) / (maxs - mins + 1e-8)

X_tr_ang = normalize_angles(X_tr_pca)
X_te_ang = normalize_angles(X_te_pca)

# ─── 4) QCNN definition ──────────────────────────────────────────────────────
n_wires = int(np.log2(PCA_COMP))  # should be 2
dev = qml.device("default.qubit", wires=n_wires)

def conv_block(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires)

def pool_block(param, wires):
    qml.CRY(param, wires=[wires[1], wires[0]])

def flatten_weights(weights):
    return pnp.concatenate([weights['conv1'], pnp.array([weights['pool1']]), weights['conv2']])

def unflatten_weights(flat):
    return {
        'conv1': flat[:2],
        'pool1': flat[2],
        'conv2': flat[3:5]
    }

@qml.qnode(dev)
def qcnn_circuit(x, flat_weights):
    weights = unflatten_weights(flat_weights)
    qml.AmplitudeEmbedding(x, wires=range(n_wires), normalize=True)
    conv_block(weights['conv1'], wires=[0,1])
    pool_block(weights['pool1'], wires=[0,1])
    conv_block(weights['conv2'], wires=[0,1])
    return qml.expval(qml.PauliZ(0))

# ─── 5) Probability mapping & metrics ─────────────────────────────────────────
def predict_prob(x, flat_weights):
    z = qcnn_circuit(x, flat_weights)
    return (z + 1.0) / 2.0

# Use pnp.* for all math in differentiable code!
def bce(y, p):
    p = pnp.clip(p, 1e-8, 1-1e-8)
    return -pnp.mean(y * pnp.log(p) + (1-y) * pnp.log(1-p))

def accuracy(y, p):
    preds = (p >= 0.5).astype(int)
    return np.mean(preds == y)

rng = np.random.default_rng(1)
init_weights = {
    'conv1': pnp.array(rng.normal(0, 0.1, size=2), requires_grad=True),
    'pool1': pnp.array(rng.normal(0, 0.1), requires_grad=True),
    'conv2': pnp.array(rng.normal(0, 0.1, size=2), requires_grad=True),
}
flat_weights = flatten_weights(init_weights)
opt = qml.AdamOptimizer(stepsize=LR)

train_loss, train_acc = [], []
print("Training QCNN…")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    idx = rng.choice(len(X_tr_ang), size=BATCH_SIZE, replace=False)
    Xb, yb = X_tr_ang[idx], y_tr[idx]
    def cost(w):
        preds = pnp.array([predict_prob(x, w) for x in Xb])
        return bce(yb, preds)
    flat_weights = opt.step(cost, flat_weights)
    probs_tr = np.array([predict_prob(x, flat_weights) for x in X_tr_ang])
    L = bce(y_tr, probs_tr)
    A = accuracy(y_tr, probs_tr)
    train_loss.append(L)
    train_acc.append(A)
    print(f"Epoch {epoch:2d} | Loss={L:.4f} | Acc={A:.3f} | Time={(time.time()-t0):.2f}s")
NEW_ROOT = os.path.join(OUTPUT_ROOT, date_str)
os.makedirs(NEW_ROOT, exist_ok=True)

plot_path = os.path.join(NEW_ROOT, f"Sid---{time_str}-graph.png")
plt.figure(figsize=(6,4))
plt.subplot(121)
plt.plot(train_loss, '-o'); plt.title("Train BCE Loss")
plt.subplot(122)
plt.plot(train_acc, '-s'); plt.title("Train Accuracy")
plt.tight_layout(); plt.savefig(plot_path); plt.close()
print(f"Saved training plots to {plot_path}")



probs_te = np.array([predict_prob(x, flat_weights) for x in X_te_ang])
print("Final Test BCE:", bce(y_te, probs_te))
print("Final Test Acc:", accuracy(y_te, probs_te))


file_path = os.path.join(NEW_ROOT, f"Sid---{time_str}.txt")
with open(file_path, "w") as f:
    f.write("Test type: Binary classification\n")
    f.write("Hyperparameters:\n")
    f.write(f"IMG_SIZE = {IMG_SIZE}\n")
    f.write(f"PCA_COMP = {PCA_COMP}\n")
    f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
    f.write(f"EPOCHS = {EPOCHS}\n")
    f.write(f"LR = {LR}\n")
    f.write("Detailed test predictions:\n")
    f.write(f"Final Test BCE: {bce(y_te, probs_te):.4f}\n")
    f.write(f"Final Test Acc: {accuracy(y_te, probs_te):.4f}\n")
    f.write(f"Maximum accuracy: {max(train_acc):.4f} at epoch {train_acc.index(max(train_acc)) + 1}\n")
print(f"Saved test outputs to {file_path}")