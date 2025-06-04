import matplotlib
matplotlib.use("TkAgg")


import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np



cartella_csv = "."
keyword = "SUPERVISED"

# Costruisce la lista dei file CSV che contengono la parola chiave nel nome
pattern = os.path.join(cartella_csv, "*.csv")
csv_files = [f for f in glob.glob(pattern) if keyword in os.path.basename(f)]

if not csv_files:
    raise FileNotFoundError(f"No file containing '{keyword}' in the folder.")

# === PLOT ===
colors = plt.cm.viridis_r(np.linspace(0, 1, len(csv_files)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    nome_file = os.path.splitext(os.path.basename(file))[0]  # estrae nome senza estensione

    # Plot accuracy vs tempo
    ax1.plot(df["time_sec"], df["accuracy"], label=nome_file, color=colors[i])

    # Plot accuracy vs epoche
    ax2.plot(df["epoch"], df["accuracy"], label=nome_file, color=colors[i])

# Configurazione dei grafici
ax1.set_title("Accuracy over time (s)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Accuracy")
ax1.grid(True)
ax1.legend()

ax2.set_title("Accuracy over Epochs")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()