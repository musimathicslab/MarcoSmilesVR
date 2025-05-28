
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns
from MSenv import MS_env, retrieve_dataset_pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_buffer import BasicBuffer

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim * 2),
            ##nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim * 2, self.input_dim),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim)
        )

        '''
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, self.output_dim)
        )
        '''

    def forward(self, state):
        qvals = self.fc(state)
        return qvals



class DQNAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000,tau=0.005):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.tau = tau
        #If CUDA available select it, else CPU mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

        #creating 2 model
        #Model that chose the actions
        self.model = DQN(48, env.action_space.n).to(self.device)
        #Target that evaluate the actions
        self.target_model = DQN(48, env.action_space.n).to(self.device)

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        if (np.random.randn() < eps):
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        # resize tensors
        actions = actions.view(actions.size(0))
        dones = dones.view(dones.size(0))

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions.view(actions.size(0), 1))
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        loss = F.mse_loss(curr_Q, expected_Q.detach())

        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)





labels,features=retrieve_dataset_pd(r"C:\Users\Daniele\Documents\GitHub\MarcoSmilesVR\GestureRecognitionModule\TestDataset12.csv")
#creating env for MarcoSmiles
env = MS_env(features,labels)



# Creating agent
trained_agent = DQNAgent(env)

# Chargin weight
trained_agent.model.load_state_dict(torch.load(r"C:\Users\Daniele\Documents\GitHub\MarcoSmilesVR\GestureRecognitionModule\utils\DQN_&_Double_DQN\HGMSD_24_2ep_DQN_model.pth"))
#set the model in evalutation mode
trained_agent.model.eval()



predicted_labels=[]
for feature,label in zip(features,labels):

    # Converti i dati in uno stato utilizzabile dal modello

    feature = torch.FloatTensor(feature).float().unsqueeze(0).to(trained_agent.device)
    print(feature)
    print(label)

    # Fai una predizione utilizzando il modello addestrato
    #predicted_qvals = trained_agent.model(feature)
    predicted_qvals = trained_agent.model(feature)


    # Ottieni l'azione predetta
    predicted_action = torch.argmax(predicted_qvals.cpu().detach()).item()

    print(f"Predicted Action: {predicted_action}")

    predicted_labels.append(predicted_action)




# Accuracy
accuracy = accuracy_score(labels, predicted_labels)
print(f'Accuracy: {accuracy}')

# Precision
precision = precision_score(labels, predicted_labels, average='macro')
print(f'Precision: {precision}')

# Recall
recall = recall_score(labels, predicted_labels, average='micro')
print(f'Recall: {recall}')

# F1 score
f1 = f1_score(labels, predicted_labels, average='micro')
print(f'F1 Score: {f1}')

# Confusion matrix
conf_matrix = confusion_matrix(labels, predicted_labels)
print(f'Confusion Matrix:\n{conf_matrix}')

# Visualizzazione della Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(12)), yticklabels=list(range(12)))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


