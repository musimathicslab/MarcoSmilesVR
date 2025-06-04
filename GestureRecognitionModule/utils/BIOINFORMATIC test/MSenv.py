import gym
import pandas as pd
import numpy as np


class MS_env(gym.Env):
    def __init__(self,features,labels):
        super(MS_env, self).__init__()

        # Observation space 48 float value (24 for each hand)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(96,), dtype=np.float32),
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


