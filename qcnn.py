#!/usr/bin/env python3
import os
import time
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer  # reverted to Nesterov
import matplotlib.pyplot as plt
from datetime import datetime
import random

# ─── Hyperparameters (set these manually) ────────────────────────────────────
DATA_ROOT     = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/ALL-DATA"
OUTPUT_ROOT   = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/outputs"
CLASS_FOLDERS = {"Glioma": 0, "Meningioma": 1, "No-Tumor": 2, "Pituitary": 3}  # CHANGED: four classes!

IMG_SIZE      = 8
PCA_COMP      = 16  
BATCH_SIZE    = 64
EPOCHS        = 400
LR            = 0.06
RANDOM_STATE  = random.randint(1, 1000)

print("RANDOM_STATE:", RANDOM_STATE)

# ─── Utility functions ──────────────────────────────────────────────────────
def load_topdown(root, folders, img_size):
    X, y = [], []
    for cls, lbl in folders.items():
        view_dir = os.path.join(root, cls, "1")
        if not os.path.isdir(view_dir):
            print(f"Missing view folder: {view_dir}")
            continue
        for fn in os.listdir(view_dir):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                img = Image.open(os.path.join(view_dir, fn)) \
                         .convert("L") \
                         .resize((img_size, img_size))
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:
                    img = img.rotate(random.choice([0, 90, 180, 270]))
                X.append(np.array(img).flatten() / 255.0)
                y.append(lbl)
    X = np.array(X); y = np.array(y)
    print(f"Loaded {len(X)} top-down images.")
    return X, y

def normalize_angles(mat):
    mins = mat.min(axis=1, keepdims=True)
    maxs = mat.max(axis=1, keepdims=True)
    return np.pi * (mat - mins) / (maxs - mins + 1e-8)

def flatten_weights(weights):
    return pnp.concatenate([
        weights['conv1'],
        pnp.array([weights['pool1']]),
        weights['conv2'],
        weights['conv3'],
        pnp.array([weights['pool2']])
    ])

def unflatten_weights(flat):
    return {
        'conv1': flat[:2],
        'pool1': flat[2],
        'conv2': flat[3:5],
        'conv3': flat[5:7],
        'pool2': flat[7]
    }

# ─── Main execution ──────────────────────────────────────────────────────────
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H-%M-%S")

# 1) Load data (with augmentation)
X, y = load_topdown(DATA_ROOT, CLASS_FOLDERS, IMG_SIZE)

# 2) Train/test split on all FOUR classes  CHANGED
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")

# 3) One-hot encode labels -- now for FOUR classes CHANGED
n_classes = 4  # ADDED
y_tr_oh = np.eye(n_classes)[y_tr]
y_te_oh = np.eye(n_classes)[y_te]

# 4) PCA & angle normalization
pca = PCA(n_components=PCA_COMP)
X_tr_pca = pca.fit_transform(X_tr)
X_te_pca = pca.transform(X_te)
print(f"PCA explained variance sum: {pca.explained_variance_ratio_.sum():.3f}")
X_tr_ang = normalize_angles(X_tr_pca)
X_te_ang = normalize_angles(X_te_pca)

# 5) QCNN definition -- outputs n_classes expvals
n_wires = int(np.log2(PCA_COMP))
dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def qcnn_circuit(x, flat_weights):
    weights = unflatten_weights(flat_weights)
    qml.AmplitudeEmbedding(x, wires=range(n_wires), normalize=True)
    # conv1
    qml.CNOT(wires=[0,1])
    qml.RY(weights['conv1'][0], wires=0)
    qml.RY(weights['conv1'][1], wires=1)
    # pool1
    qml.CRY(weights['pool1'], wires=[1,0])
    # conv2
    qml.CNOT(wires=[0,1])
    qml.RY(weights['conv2'][0], wires=0)
    qml.RY(weights['conv2'][1], wires=1)
    # conv3
    qml.CNOT(wires=[0,1])
    qml.RY(weights['conv3'][0], wires=0)
    qml.RY(weights['conv3'][1], wires=1)
    # pool2
    qml.CRY(weights['pool2'], wires=[1,0])
    # NOW return n_classes (4) outputs
    return [qml.expval(qml.PauliZ(i)) for i in range(n_classes)]  # CHANGED

def softmax(z):
    z_max = pnp.max(z, axis=1, keepdims=True)
    e = pnp.exp(z - z_max)
    return e / pnp.sum(e, axis=1, keepdims=True)

def cce(y_oh, probs):
    return -pnp.mean(pnp.sum(y_oh * pnp.log(probs + 1e-8), axis=1))

def accuracy_mc(y_true, probs):
    return np.mean(np.argmax(probs, axis=1) == y_true)

# Initialize weights & Nesterov optimizer
rng = np.random.default_rng(RANDOM_STATE)
init_weights = {
    'conv1': pnp.array(rng.normal(0, 0.1, size=2), requires_grad=True),
    'pool1': pnp.array(rng.normal(0, 0.1),     requires_grad=True),
    'conv2': pnp.array(rng.normal(0, 0.1, size=2), requires_grad=True),
    'conv3': pnp.array(rng.normal(0, 0.1, size=2), requires_grad=True),
    'pool2': pnp.array(rng.normal(0, 0.1),     requires_grad=True),
}
flat_weights = flatten_weights(init_weights)
opt = NesterovMomentumOptimizer(stepsize=LR, momentum=0.9)

# 6) Training loop
train_loss, train_acc = [], []
print("Training QCNN…")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    idx = rng.choice(len(X_tr_ang), size=BATCH_SIZE, replace=False)
    Xb, yb_oh = X_tr_ang[idx], y_tr_oh[idx]

    def cost(w):
        raws = pnp.stack([qcnn_circuit(x, w) for x in Xb])
        return cce(yb_oh, softmax(raws))

    flat_weights = opt.step(cost, flat_weights)

    raws_tr = pnp.stack([qcnn_circuit(x, flat_weights) for x in X_tr_ang])
    probs_tr = softmax(raws_tr)
    L = cce(y_tr_oh, probs_tr)
    A = accuracy_mc(y_tr, probs_tr)
    train_loss.append(L); train_acc.append(A)

    print(f"Epoch {epoch:3d} | Loss={L:.4f} | Acc={A:.3f} | Time={(time.time()-t0):.2f}s")

# 7) Save training curves
NEW_ROOT = os.path.join(OUTPUT_ROOT, date_str)
os.makedirs(NEW_ROOT, exist_ok=True)
plot_path = os.path.join(NEW_ROOT, f"Sid---{time_str}-graph.png")
plt.figure(figsize=(6,4))
plt.subplot(121); plt.plot(train_loss, '-o'); plt.title("Train CCE Loss")
plt.subplot(122); plt.plot(train_acc, '-s'); plt.title("Train Accuracy")
plt.tight_layout(); plt.savefig(plot_path); plt.close()
print(f"Saved training plots to {plot_path}")

# 8) Final evaluation
raws_te = pnp.stack([qcnn_circuit(x, flat_weights) for x in X_te_ang])
probs_te = softmax(raws_te)
test_cce = cce(y_te_oh, probs_te)
test_acc = accuracy_mc(y_te, probs_te)
print("Final Test CCE:", test_cce)
print("Final Test Acc:", test_acc)

# 9) Save test metrics
file_path = os.path.join(NEW_ROOT, f"Sid---{time_str}.txt")
with open(file_path, "w") as f:
    f.write("Hyperparameters:\n")
    f.write(f"IMG_SIZE      = {IMG_SIZE}\n")
    f.write(f"PCA_COMP      = {PCA_COMP}\n")
    f.write(f"BATCH_SIZE    = {BATCH_SIZE}\n")
    f.write(f"EPOCHS        = {EPOCHS}\n")
    f.write(f"LR            = {LR}\n")
    f.write(f"RANDOM_STATE  = {RANDOM_STATE}\n\n")
    f.write("Detailed test predictions:\n")
    f.write(f"Final Test CCE: {test_cce:.4f}\n")
    f.write(f"Final Test Acc: {test_acc:.4f}\n")
    f.write(f"Maximum training Acc: {max(train_acc):.4f} at epoch {train_acc.index(max(train_acc))+1}\n")
print(f"Saved test outputs to {file_path}")
