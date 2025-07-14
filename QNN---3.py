#!/usr/bin/env python3
"""
10-Qubit QNN on Brain Tumor MRI Dataset
- 32×32 images → 1024 dims → PCA → 10 features
- Data encoding: RY angle rotations
- Variational ansatz: RealAmplitudes (reps=3)
- Statevector analytic probabilities
- Scatterplots of PCA components
- Training with SciPy COBYLA optimizer + mini-batch updates
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Qiskit imports
try:
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import Statevector
    print(f"Using Qiskit version: {qiskit.__version__}")
except ImportError:
    print("ERROR: Qiskit not found. Please activate the environment with Qiskit installed (pip install qiskit qiskit-aer).")
    sys.exit(1)

# Hyperparameters
DATA_ROOT      = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantom-Nural-Network-2/Dataset-vs-CNN"
CLASS_FOLDERS  = {"Glioma-Backup":0, "Meningioma-Backup":1, "Pituitary-Backup":2}
BINARY_CLASSES = [0,1]
IMG_SIZE       = 32        # 32×32
N_QUBITS       = 10        # 2^10=1024 amplitudes
PCA_COMPONENTS = N_QUBITS  # reduce to 10 features
REPS           = 3         # ansatz depth
TEST_SPLIT     = 0.1       # split fraction
EPOCHS         = 10        # training epochs
TRAIN_LIMIT    = 1000      # limit for speed
MAXITER        = 200       # optimizer max iterations per epoch
BATCH_SIZE     = 100       # mini-batch size

# 1) Load & preprocess images
def load_data_topdown_only(root, class_folders):
    X, y = [], []
    for cls_name, label in class_folders.items():
        view_dir = os.path.join(root, cls_name, "1")   # only the “1” folder
        if not os.path.isdir(view_dir):
            print(f"Warning: missing {view_dir}")
            continue
        for fname in os.listdir(view_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img = (
                Image.open(os.path.join(view_dir, fname))
                     .convert("L")
                     .resize((IMG_SIZE, IMG_SIZE))
            )
            X.append(np.array(img).flatten() / 255.0)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} top-down images from view “1” only")
    return X, y


print("=== Loading Data ===")
X_all, y_all = load_data_topdown_only(DATA_ROOT, CLASS_FOLDERS)
# filter classes
mask = np.isin(y_all, BINARY_CLASSES)
X_all, y_all = X_all[mask], y_all[mask]
y_all = (y_all == BINARY_CLASSES[1]).astype(int)
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=TEST_SPLIT, random_state=42, stratify=y_all)
X_tr_raw, y_tr = X_tr_raw[:TRAIN_LIMIT], y_tr[:TRAIN_LIMIT]
print(f"Train: {X_tr_raw.shape}, Test: {X_te_raw.shape}")

# 2) PCA dimensionality reduction
print("=== PCA Reduction ===")
pca = PCA(n_components=PCA_COMPONENTS)
pca.fit(np.vstack([X_tr_raw, X_te_raw]))
X_tr_pca = pca.transform(X_tr_raw)
X_te_pca = pca.transform(X_te_raw)
print(f"Explained variance sum: {pca.explained_variance_ratio_.sum():.3f}")

# PCA scatterplots
def plot_pca(X_pca, y, title):
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='tab10', legend='full')
    plt.title(title); plt.show()
plot_pca(X_tr_pca, y_tr, 'Train PCA')
plot_pca(X_te_pca, y_te, 'Test PCA')

# 3) Data encoding: RY rotations
def angle_encode(vec):
    v = (vec - vec.min())/(vec.max() - vec.min() + 1e-8)
    return v * np.pi
X_tr_enc = [angle_encode(v) for v in X_tr_pca]
X_te_enc = [angle_encode(v) for v in X_te_pca]

# 4) Build variational ansatz
ansatz = RealAmplitudes(num_qubits=N_QUBITS, reps=REPS, entanglement='linear')
param_syms = list(ansatz.parameters)
print(f"Ansatz: qubits={N_QUBITS}, reps={REPS}, params={len(param_syms)}")

# 5) Statevector-based prediction
def predict(params, feats):
    qc = QuantumCircuit(N_QUBITS)
    for i,a in enumerate(feats): qc.ry(a, i)
    qc.barrier()
    qc.compose(ansatz.assign_parameters(dict(zip(param_syms,params))), range(N_QUBITS), inplace=True)
    state = Statevector.from_label('0'*N_QUBITS)
    final = state.evolve(qc)
    probs = final.probabilities_dict()
    return sum(p for b,p in probs.items() if b[0]=='1')

# 6) Loss, accuracy, and mini-batch cost
def bce(y, p, eps=1e-8): p=np.clip(p,eps,1-eps); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
def acc(y, p): return np.mean((p>=0.5).astype(int)==y)
def cost(params, X, y):
    idx = np.random.choice(len(X), BATCH_SIZE, replace=False)
    Xb = [X[i] for i in idx]; yb=y[idx]
    preds = np.array([predict(params,x) for x in Xb])
    return bce(yb, preds)

# 7) Training loop (COBYLA optimizer)
print("=== Training with COBYLA + mini-batch ===")
params = np.random.randn(len(param_syms))
hist = {'loss':[], 'acc':[]}
for ep in range(1, EPOCHS+1):
    print(f"Epoch {ep}/{EPOCHS}")
    t0 = time.time()
    res = minimize(lambda p: cost(p, X_tr_enc, y_tr), params,
                   method='COBYLA', options={'maxiter':MAXITER})
    params = res.x
    print(f"  Done in {time.time()-t0:.1f}s, fun={res.fun:.4f}")
    preds_full = np.array([predict(params,x) for x in X_tr_enc])
    loss_full = bce(y_tr, preds_full)
    acc_full  = acc(y_tr, preds_full)
    print(f"  Train full BCE={loss_full:.4f}, ACC={acc_full:.3f}")
    hist['loss'].append(loss_full)
    hist['acc'].append(acc_full)

# 8) Plot training curves
plt.figure(figsize=(6,4))
plt.plot(hist['loss'], marker='o', label='BCE Loss')
plt.plot(hist['acc'],  marker='s', label='Accuracy')
plt.xlabel('Epoch'); plt.legend(); plt.title('Training Curves')
plot_path = os.path.join(output_dir, f"{time_str}-image.png")
plt.savefig(plot_path)

# 9) Final evaluation on test set
preds_test = np.array([predict(params,x) for x in X_te_enc])
print(f"Test BCE={bce(y_te,preds_test):.4f}, ACC={acc(y_te,preds_test):.3f}")
