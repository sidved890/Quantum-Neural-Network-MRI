import os, re
import pandas as pd
from datetime import datetime

# regex to find a timestamp of the form 09-42-15 anywhere in the filename
time_re = re.compile(r'(\d{2}-\d{2}-\d{2})')

now      = datetime.now()
date_str = now.strftime("%Y-%m-%d")     # e.g. “2025-07-14”

root_dir = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/outputs"
rows = []

for date in os.listdir(root_dir):
    date_path = os.path.join(root_dir, date)
    if not os.path.isdir(date_path):
        continue

    for fname in os.listdir(date_path):
        if not fname.lower().endswith(".txt"):
            continue

        full_path = os.path.join(date_path, fname)
        content = open(full_path).read()

        # extract timestamp if present
        m = time_re.search(fname)
        time_str = m.group(1) if m else "unknown"

        entry = {
            "date": date,
            "time": time_str,
        }

        # your existing parsing logic
        for pm in re.finditer(r'(\w+)\s*=\s*([0-9.]+)', content):
            entry[pm.group(1)] = float(pm.group(2))
        if m2 := re.search(r'Final Test BCE:\s*([0-9.]+)', content):
            entry["Final_Test_BCE"] = float(m2.group(1))
        if m3 := re.search(r'Final Test Acc:\s*([0-9.]+)', content):
            entry["Final_Test_Acc"] = float(m3.group(1))
        if m4 := re.search(r'Maximum accuracy:\s*([0-9.]+)\s*at epoch\s*(\d+)', content):
            entry["Max_Accuracy"]  = float(m4.group(1))
            entry["Max_Acc_Epoch"] = int(m4.group(2))

        rows.append(entry)

df = pd.DataFrame(rows)

# === write out the Excel ===
base_sheets_dir = os.path.join(root_dir, "sheets")
save_dir        = os.path.join(base_sheets_dir, date_str)
os.makedirs(save_dir, exist_ok=True)

now_time = datetime.now().strftime("%H-%M-%S")
out_path = os.path.join(save_dir, f"Sid---{now_time}.xlsx")
df.to_excel(out_path, index=False)
print(f"Written spreadsheet to {out_path}")
