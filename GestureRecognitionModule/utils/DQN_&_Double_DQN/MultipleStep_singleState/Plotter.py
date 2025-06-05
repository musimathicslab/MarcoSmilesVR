import matplotlib
matplotlib.use("TkAgg")


import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def extract_stats(df, nome_file):
    """Estrae max accuracy, epoca, timestamp e media per un singolo DataFrame."""
    max_accuracy = df["accuracy"].max()
    idx_max = df["accuracy"].idxmax()
    epoch_max = df.loc[idx_max, "epoch"]
    time_max = df.loc[idx_max, "time_sec"]
    media_accuracy = df["accuracy"].mean()

    return [
        nome_file,
        f"{max_accuracy:.4f}",
        f"{epoch_max}",
        f"{time_max:.2f}s",
        f"{media_accuracy:.4f}"
    ]

def split_name(nome_file):
    parts = nome_file.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1]), parts[-1]
    else:
        return nome_file, nome_file

def save_stats(statistiche, titolo_tabella=""):
    output_image = f"statistics_summary_{titolo_tabella}.png"
    colonne = ["Batch", "Max Accuracy", "Epoca", "Timestamp", "Media Accuracy"]

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(statistiche) + 1.5))
    ax.axis("off")
    tabella = ax.table(cellText=statistiche, colLabels=colonne, loc="center", cellLoc="center")
    tabella.auto_set_font_size(False)
    tabella.set_fontsize(10)
    tabella.scale(1.2, 1.2)

    # Allarga prima colonna
    for key, cell in tabella.get_celld().items():
        row, col = key
        if col == 0:
            cell.set_width(0.4)
        else:
            cell.set_width(0.15)

    plt.title(f"Statistiche per {titolo_tabella}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Tabella salvata come: {output_image}")


cartella_csv = "."
keyword = "NO_Target_NO"
stats=[]
# Costruisce la lista dei file CSV che contengono la parola chiave nel nome
pattern = os.path.join(cartella_csv, "*.csv")
csv_files = [f for f in glob.glob(pattern) if keyword in os.path.basename(f)]

if not csv_files:
    raise FileNotFoundError(f"No file containing '{keyword}' in the folder.")

# === PLOT ===
colors = plt.cm.viridis_r(np.linspace(0, 1, len(csv_files)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


prefix_comune = None
for i, file in enumerate(csv_files):
    df = pd.read_csv(file)
    nome_file = os.path.splitext(os.path.basename(file))[0]  # estrae nome senza estensione

    prefix, suffix = split_name(nome_file)
    if prefix_comune is None:
        prefix_comune = prefix
    # Plot accuracy vs tempo
    ax1.plot(df["time_sec"], df["accuracy"], label=nome_file, color=colors[i])

    # Plot accuracy vs epoche
    ax2.plot(df["epoch"], df["accuracy"], label=nome_file, color=colors[i])
    stats.append(extract_stats(df,suffix))
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
fig.suptitle(f"Risultati per {prefix_comune}", fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # lascia spazio per il titolo
plt.show()

save_stats(stats,titolo_tabella=prefix_comune)
