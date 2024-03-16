import pandas as pd

def retrieve_dataset_pd(path_file):
    path = path_file
    df = pd.read_csv(path, header=None)
    return df

# Carica il dataset
rows = retrieve_dataset_pd("dataset path to reduce")

rows_number = rows.shape[0]
print(rows_number)
# Lista degli indici delle righe da selezionare
selected_indexes = []
temp=int(rows_number/500)
for i in range(temp):
    selected_indexes.append(500*i)

# Seleziona le righe corrispondenti agli indici specificati
selected_rows = rows.iloc[selected_indexes]

print(selected_rows)
selected_rows.drop(0)

selected_rows.to_csv("result dataset path", index=False,header=False)
