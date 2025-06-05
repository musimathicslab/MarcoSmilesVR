import gym
import pandas as pd
import numpy as np
import torch

# === PARAMETRI DI CODIFICA ===
char_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
base = 4
one_hot_size = 24  # 6 lettere * 4 posizioni (esamero in base-4)



class MS_env(gym.Env):
    def __init__(self,features,labels):
        super(MS_env, self).__init__()

        # Observation space 48 float value (24 for each hand)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(2304,), dtype=np.float32),
        ))

        #Action discrete (select note)
        #self.action_space = gym.spaces.Discrete(25)  #(0 to 24)

        self.action_space = gym.spaces.Discrete(2)

        # index used to chose the next feature to train+
        self.features_index = 0
        self.label_array = labels
        self.features = features
        self.actual_label=self.label_array[self.features_index]

        self.state = (self.features[self.features_index])



    def reset(self):
        #In each episode we consider a single position of hands

        if(self.features_index==len(self.features)):
            self.features_index=0
        self.state = self.features[self.features_index]
        self.actual_label=self.label_array[self.features_index]
        self.features_index += 1
        return self.state

    def step(self, action):
        done = False
        if(action==self.actual_label):
            #print("note reached "+str(self.actual_label))
            reward = 100
            done = True

        else:
            reward = 0
            done=True

       
        return self.state, reward, done, {}

    def render(self, mode='human'):
       
        pass

    def get_correct_action(self):
        return self.actual_label

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


# === 1. Funzione di codifica one-hot ===
def hexamer_to_one_hot(hexamer):
    one_hot = [0] * one_hot_size
    for i, char in enumerate(hexamer):
        val = char_map.get(char, 0)  # default 0
        one_hot[i * 4 + val] = 1
    return one_hot

# === 2. Legge il file TSV e crea tensori ===
def load_data_from_tsv(tsv_path):
    features = []
    labels = []

    with open(tsv_path, 'r') as f:
        header = f.readline().strip().split('\t')  # salta intestazione
        for line in f:
            row = dict(zip(header, line.strip().split('\t')))
            label = int(row['label'])
            hexamers = row['sequence'].strip().split()

            encoded = []
            for h in hexamers:
                onehot = hexamer_to_one_hot(h)
                encoded.extend(onehot)

            features.append(encoded)
            labels.append(label)

    # Padding (tutte le sequenze devono avere la stessa lunghezza)
    max_len = max(len(f) for f in features)
    for i in range(len(features)):
        while len(features[i]) < max_len:
            features[i].append(0)

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y
