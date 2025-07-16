import pandas as pd
import os, glob
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from itertools import product

sheets_root = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/outputs/sheets"

# 1) find every .xlsx in any sub-folder, then pick the one with the newest modification time
all_excels = glob.glob(os.path.join(sheets_root, "*", "*.xlsx"))
latest    = max(all_excels, key=os.path.getmtime)

print("Loading latest sheet:", latest)
df = pd.read_excel(latest)

# Drop rows missing our new target
before = len(df)
df = df.dropna(subset=["Final_Test_Acc"])
after  = len(df)
print(f"Dropped {before-after} rows with missing Final_Test_Acc, {after} remain.")

# Inputs and new target
X = df[["IMG_SIZE","PCA_COMP","BATCH_SIZE","EPOCHS","LR"]].values
y = df["Final_Test_Acc"].values

# Build degree‑2 polynomial features
poly = PolynomialFeatures(degree=2, include_bias=True)
X2 = poly.fit_transform(X)

# Fit a linear model
model = LinearRegression()
model.fit(X2, y)

# Print formula
names     = poly.get_feature_names_out(["IMG_SIZE","PCA_COMP","BATCH_SIZE","EPOCHS","LR"])
coef      = model.coef_
intercept = model.intercept_

print("Final_Test_Acc ≈\n    {:.5f}".format(intercept), end="")
for c, n in sorted(zip(coef, names), key=lambda x: -abs(x[0]))[1:]:
    print(f"\n  + ({c:+.5f}) × {n}", end="")
print()

# Evaluate on your tried grid
pred     = model.predict(X2)
best_idx = np.argmax(pred)
print(f"\nOn your tried grid, highest predicted Final_Test_Acc = {pred[best_idx]:.4f}")
print("  at hyperparams:", dict(
    zip(["IMG_SIZE","PCA_COMP","BATCH_SIZE","EPOCHS","LR"], X[best_idx])
))

# Simple continuous search around that point
best   = X[best_idx]
ranges = {
    "IMG_SIZE":   [best[0]],
    "PCA_COMP":   [best[1]],
    "BATCH_SIZE": [best[2]],
    "EPOCHS":     np.linspace(best[3]*0.5, best[3]*1.5, 5),
    "LR":         np.linspace(best[4]*0.5, best[4]*1.5, 5),
}
grid = np.array([
    [i,j,k,l,m]
    for i,j,k,l,m in product(
        ranges["IMG_SIZE"], ranges["PCA_COMP"],
        ranges["BATCH_SIZE"], ranges["EPOCHS"],
        ranges["LR"]
    )
])
p2 = model.predict(poly.transform(grid))
ix = np.argmax(p2)
print(f"\nBest continuous suggestion: {p2[ix]:.4f}")
print("  hyperparams:", dict(zip(
    ["IMG_SIZE","PCA_COMP","BATCH_SIZE","EPOCHS","LR"], grid[ix]
)))
