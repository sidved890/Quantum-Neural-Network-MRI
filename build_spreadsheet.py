import os, re
import pandas as pd

# 1) Point this at the folder that holds all your {date}/ subfolders
root_dir = "/Users/siddharthvedam/Downloads/Quantum-Neural-Network-MRI-1/outputs"

# 2) Prepare regexes
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
# 3) Write out an Excel file
df.to_excel("experiment_results.xlsx", index=False)
print("Written experiment_results.xlsx — open it in Excel or Sheets!")
