import os, re
import pandas as pd
from datetime import datetime

now      = datetime.now()
date_str = now.strftime("%Y-%m-%d")     # e.g. “2025-07-14”
time_str = now.strftime("%H-%M-%S")     # e.g. “09-42-15”

# 1) Point this at the folder that holds all your {date}/ subfolders
<<<<<<< HEAD
root_dir = "/Users/siddharthvedam/Downloads/Track 7---SRA/Quantum-Neural-Network-MRI/outputs"
=======
root_dir = ""
>>>>>>> 21b02d2 (Version for Rui)

param_re    = re.compile(r'(\w+)\s*=\s*([0-9.]+)')
bce_re      = re.compile(r'Final Test BCE:\s*([0-9.]+)')
acc_re      = re.compile(r'Final Test Acc:\s*([0-9.]+)')
max_acc_re  = re.compile(r'Maximum accuracy:\s*([0-9.]+)\s*at epoch\s*(\d+)')

rows = []
for date in os.listdir(root_dir):
    date_path = os.path.join(root_dir, date)
    if not os.path.isdir(date_path): continue

    for fname in os.listdir(date_path):
        if not fname.endswith(".txt"): continue
        time = fname[:-4]
        time = time[-8:]
        content = open(os.path.join(date_path, fname)).read()

        entry = {"date": date, "time": time}
        for m in param_re.finditer(content):
            entry[m.group(1)] = float(m.group(2))
        if m := bce_re.search(content):      entry["Final_Test_BCE"] = float(m.group(1))
        if m := acc_re.search(content):      entry["Final_Test_Acc"] = float(m.group(1))
        if m := max_acc_re.search(content):
            entry["Max_Accuracy"]    = float(m.group(1))
            entry["Max_Acc_Epoch"]   = int(m.group(2))

        rows.append(entry)
df = pd.DataFrame(rows)

# === new: prepare the save directory ===
base_sheets_dir = os.path.join(root_dir, "sheets")
save_dir        = os.path.join(base_sheets_dir, date_str)
os.makedirs(save_dir, exist_ok=True)

# filename as Sid---{time}.xlsx
filename = f"Sid---{time_str}.xlsx"
out_path = os.path.join(save_dir, filename)

# 3) Write out the Excel file there
df.to_excel(out_path, index=False)
<<<<<<< HEAD
print(f"Written spreadsheet to {out_path}")
=======
print(f"Written spreadsheet to {out_path}")
>>>>>>> 21b02d2 (Version for Rui)
