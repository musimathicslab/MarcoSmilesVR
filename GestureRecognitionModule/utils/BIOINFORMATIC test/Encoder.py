import csv
import time
# Mappa simboli → cifre base-4
char_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
base = 4


# Funzione per convertire un esamero in numero base-4
def hexamer_to_base4_value(hexamer):
    value = 0
    for i, char in enumerate(hexamer):
        digit = char_map.get(char, 0)  # default 0 per caratteri non riconosciuti
        power = len(hexamer) - i - 1
        value += digit * (base ** power)
    return float(value)


# Funzione principale
def process_tsv(input_tsv, output_csv):
    with open(input_tsv, 'r', newline='') as infile:
        reader = csv.DictReader(infile, delimiter='\t')

        # Preparazione output: calcolo numero massimo di esameri
        all_rows = list(reader)
        max_hexamers = max(len(row['sequence'].strip().split()) for row in all_rows)

        # Intestazione CSV


        with open(output_csv, 'w', newline='') as outfile:
            writer = csv.writer(outfile)



            for row in all_rows:
                sequence = row['sequence'].strip()

                label = row['label']
                hexamers = sequence.split()
                encoded = [hexamer_to_base4_value(h) for h in hexamers]

                # Padding per righe con meno esameri
                while len(encoded) < max_hexamers:
                    encoded.append(0.0)
                #print([label] + encoded)
                #print(sequence)
                writer.writerow([label] + encoded)
                #time.sleep(4)

# Usa uno di questi formati per il percorso
train_tsv = r'C:\Users\ACER\Documents\GitHub\MarcoSmilesVR\GestureRecognitionModule\utils\BIOINFORMATIC test\train.tsv'
# Oppure:
# train_tsv = 'C:/Users/ACER/Documents/GitHub/MarcoSmilesVR/GestureRecognitionModule/utils/BIOINFORMATIC test/train.tsv'

outputpath = "train_csv.csv"
process_tsv(train_tsv, outputpath)