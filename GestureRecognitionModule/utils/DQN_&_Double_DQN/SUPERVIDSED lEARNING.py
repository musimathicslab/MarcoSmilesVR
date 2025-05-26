import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from time import sleep

# --- Dataset sintetico ---
class IncrementalDataset(Dataset):
    def __init__(self, X=None, y=None):
        self.X = torch.tensor(X, dtype=torch.float32) if X is not None else torch.empty((0, 48))
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else torch.empty((0,), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def add_data(self, new_X, new_y):
        self.X = torch.cat((self.X, new_X))
        self.y = torch.cat((self.y, new_y))

# --- Rete semplice ---
class SimpleNet(nn.Module):
    def __init__(self, input_dim=48, num_classes=12):
        super().__init__()

        self.input_dim= input_dim
        self.output_dim= num_classes
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim * 2, self.input_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- Accuracy per classe ---
def accuracy_per_class(model, dataloader, device, num_classes):
    model.eval()
    correct = [0 for _ in range(num_classes)]
    total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # --- Stampa etichette vere e predette ---
            print("True labels:     ", labels.cpu().numpy())
            print("Predicted labels:", preds.cpu().numpy())
            print("-" * 50)

            for c in range(num_classes):
                mask = (labels == c)
                correct[c] += (preds[mask] == c).sum().item()
                total[c] += mask.sum().item()

    acc = {c: (correct[c] / total[c]) if total[c] > 0 else 0.0 for c in range(num_classes)}
    return acc


def retrieve_dataset_pd(path_file):
    path = path_file
    # print(self.current_step)
    df = pd.read_csv(path, header=None)

    df.columns = ["label"] + [f"feature_{i}" for i in range(1, len(df.columns))]

    # Estrai le etichette (y) e le features (X) come array NumPy
    y = df["label"].values
    X = df.drop("label", axis=1).values
    print(type(y))

    return y,X

# --- Main training ---
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y, x = retrieve_dataset_pd(r"C:\Users\Daniele\Documents\GitHub\MarcoSmilesVR\GestureRecognitionModule\HGMSD_12.csv")
    full_dataset = IncrementalDataset(x, y)

    streaming_dataset = IncrementalDataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(input_dim=48, num_classes=12).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    step_size = 500
    num_epochs = 1
    batch_size = 64
    total_samples = len(full_dataset)

    for tick in range(0, total_samples, step_size):
        print(f"\n--- TICK {tick // step_size + 1} ---")

        # Estrai nuovi dati
        end = min(tick + step_size, total_samples)
        new_X = full_dataset.X[tick:end]
        new_y = full_dataset.y[tick:end]

        # Aggiungi al dataset incrementale
        streaming_dataset.add_data(new_X, new_y)

        # Dataloader
        train_loader = DataLoader(streaming_dataset, batch_size=batch_size, shuffle=True)

        # Training
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

        # Accuracy per classe
        acc = accuracy_per_class(model, train_loader, device, num_classes=12)
        print("Accuracy per classe:")
        for c in sorted(acc.keys()):
            print(f"  Classe {c:02d}: {acc[c]:.2%}")


if __name__ == "__main__":
    main()
