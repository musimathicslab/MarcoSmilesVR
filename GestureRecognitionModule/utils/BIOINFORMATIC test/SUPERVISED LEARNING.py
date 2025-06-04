import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time

from time import sleep
from sklearn.metrics import accuracy_score


log_data=[]
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
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim * 2, self.input_dim*4),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim*4, self.input_dim * 2),
            #nn.Dropout(0.2),
            #nn.ReLU(),
            nn.Linear(self.input_dim*2 , self.output_dim)
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
            #print("True labels:     ", labels.cpu().numpy())
            #print("Predicted labels:", preds.cpu().numpy())
            #print("-" * 50)

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
    #print(type(y))

    return y,X


def valuating_phase(model, path_file,device,epoch,elapsed_time):
    y_val,x_val = retrieve_dataset_pd(path_file)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val).to(device)
    print(device)
    model.eval()

    with torch.no_grad():
        outputs = model(x_val)
        predicted = torch.argmax(outputs, 1)
        acc = accuracy_score(y_val.cpu().numpy(), predicted.cpu().numpy())

        # Estrazione della parte finale del path_file (nome file con estensione)
        file_name = os.path.basename(path_file)

        result_string = f"ACCURACY ON {file_name}: {acc}\n"

        # Salvataggio in append in un file di testo (ad esempio "results.txt")
        with open("results.txt", "a") as f:
            f.write(result_string)

        log_data.append({
            "epoch": epoch + 1,
            "accuracy": acc,
            "time_sec": elapsed_time
        })




def end_timer(start_time):
    """
    Prende il tempo di inizio e calcola il tempo trascorso fino ad ora.
    Restituisce la durata formattata in h, m, s e in secondi.
    """
    print(start_time)

    end_time = time.time()
    print(end_time)
    elapsed = end_time - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Tempo trascorso: {hours}h {minutes}m {seconds}s")
    return elapsed

def save_training_log_csv(df_log: pd.DataFrame, train_dataset_path: str, batchsize: int):

    # Ottieni il nome dello script Python corrente, senza estensione
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # Ottieni il nome del dataset di training, senza estensione
    train_dataset_name = os.path.splitext(os.path.basename(train_dataset_path))[0]

    # Costruisci il nome finale del CSV
    csv_filename = f"{script_name}_{train_dataset_name}_batch{batchsize}.csv"

    # Salva il DataFrame in CSV
    df_log.to_csv(csv_filename, index=False)

    print(f"CSV salvato: {csv_filename}")



# --- Main training ---
def main(batchsize, epochs,retry_final):
    start_time = time.time()

    dataset_path= r"C:\Users\Daniele\Documents\GitHub\MarcoSmilesVR\GestureRecognitionModule\HGMSD_12.csv"
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y, x = retrieve_dataset_pd(dataset_path)
    full_dataset = IncrementalDataset(x, y)

    streaming_dataset = IncrementalDataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(input_dim=48, num_classes=12).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    step_size = 500
    num_epochs = epochs
    batch_size = batchsize
    total_samples = len(full_dataset)
    print(total_samples)
    for tick in range(0, total_samples, step_size):
        starting_time = time.time()
        print(f"\n--- TICK {tick // step_size + 1} ---")


        # Estrai nuovi dati
        end = min(tick + step_size, total_samples)
        new_X = full_dataset.X[tick:end]
        new_y = full_dataset.y[tick:end]

        # Aggiungi al dataset incrementale
        streaming_dataset.add_data(new_X, new_y)

        # Dataloader
        train_loader = DataLoader(streaming_dataset, batch_size=batch_size, shuffle=True)
        '''
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
        '''
        if (tick // step_size) + 1==12:
            for i in range(retry_final):
                train_loader = DataLoader(streaming_dataset, batch_size=batch_size, shuffle=True)
                print("redoiing training : retry ",i)
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
                valuating_phase(model,r"C:\Users\Daniele\Documents\GitHub\MarcoSmilesVR\GestureRecognitionModule\TestDataset12.csv",device,i,time.time()-start_time)

    print(log_data)
    df_log = pd.DataFrame(log_data)
    print(df_log)
    save_training_log_csv(df_log,dataset_path,batchsize)



    end_timer(starting_time)
    #valuating_phase(model,r"C:\Users\Daniele\Documents\GitHub\MarcoSmilesVR\GestureRecognitionModule\HGMSD_12.csv",device)



if __name__ == "__main__":
    '''
    batchsizes = [128, 256, 512]
    epochs_list = [50, 100, 200]
    retry_finals = [5, 10, 15]

    for bs in batchsizes:
        for ep in epochs_list:
            for rf in retry_finals:
                heading = f"batchsize={bs}; epochs={ep}; retry_final={rf}"
                print(heading)

                start_time = time.time()
                main(batchsize=bs, epochs=ep, retry_final=rf)
                elapsed = time.time() - start_time

                time_str = f"Elapsed time: {elapsed:.2f} seconds"
                print(time_str)
                print()  # linea vuota per separazione

                with open("results.txt", "a") as f:
                    f.write(heading + "\n")
                    f.write(time_str + "\n\n")
    '''
    start_time = time.time()

    batchsizes = [64, 128, 256, 512]
    for bs in batchsizes:
        log_data = []
        main(batchsize=bs, epochs=10, retry_final=500)


    elapsed = time.time() - start_time

    time_str = f"Elapsed time: {elapsed:.2f} seconds"
    print(time_str)
    print()  # linea vuota per separazione