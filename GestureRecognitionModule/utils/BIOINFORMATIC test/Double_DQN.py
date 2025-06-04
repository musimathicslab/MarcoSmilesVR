from MSenv import MS_env, retrieve_dataset_pd
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gym
import torch
from basic_buffer import BasicBuffer
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals



class DQNAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, buffer_size=10000,tau=0.005):
        self.env = env
        # Learning rate
        self.learning_rate = learning_rate
        #discount factor
        self.gamma = gamma
        #that is ht edimension of the expreience replay
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        # Updating factor, is a wey to weight the target net update
        #in this script the periodical update of the net is applyed for every 1000 interval
        # if u want to use the same chose as van Hasselt, that use the periodic UPDATE u can set the tau to 1, so u will obtain a perfect copy of the online network
        # u can always try to do a mix of them using the tau and the periodic UPDATE
        self.tau = tau
        self.tau=1




        self.target_update_interval = 100

        self.update_counter = 0

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
        #print(batch)
        #time.sleep(100)
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        # resize tensors
        actions = actions.view(actions.size(0))
        dones = dones.view(dones.size(0))

        curr_Q = self.model.forward(states).gather(1, actions.view(actions.size(0), 1))
        # evaluate all the possible action starting from the next_stete associated with a q_value
        max_action_next_Q_online = self.model.forward(next_states)
        # Now i need to take the action with the max q-Value so i can doo argmax
        max_action_next_Q_online = torch.argmax(max_action_next_Q_online, 1)
        max_action_next_Q_online= max_action_next_Q_online.view(max_action_next_Q_online.size(0), 1)

        # Calculate q-values for the next states using target network
        next_Q_target = self.target_model.forward(next_states)

        # Use the target network for evaluate  the action selected by online net
        next_Q_target = next_Q_target.gather(1, max_action_next_Q_online)




        expected_Q = rewards + (1 - dones) * self.gamma * next_Q_target

        #print("THAT IS EXPECTED Q")
        #print(expected_Q)
        



        loss = F.mse_loss(curr_Q, expected_Q.detach())

        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #self.update_target_net()

        if self.update_counter % self.target_update_interval == 0:
            self.update_target_net()
            self.update_counter = 0
        else:
            self.update_counter += 1




    #Defining the function to update the target network, done every update interval
    def update_target_net(self):
        #print("updating target")
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)



def plot_rewards_line(epoch_rewards):
    plt.figure(figsize=(10, 6))

    for epoch, rewards in enumerate(epoch_rewards):
        plt.plot(rewards, label=f'Epoca {epoch + 1}', marker='o')

    plt.title('Reward per Epoca')
    plt.xlabel('Epoca')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

#retrieving features and labels
labels,features=retrieve_dataset_pd("Your train dataset path")
#creating env for MarcoSmiles
env = MS_env(features,labels)



agent= DQNAgent(env)

#set to len feature in oprder to make sure to check all the dataset for each position

episodes = len(features)


batch_size = 64
epocs = 20
max_steps=10
epoch_rewards=[]
episode_rewards =[]
print("================================================")
print(r"""
          _____                   _____          
         /\    \                 /\    \         
        /::\____\               /::\    \        
       /::::|   |              /::::\    \       
      /:::::|   |             /::::::\    \      
     /::::::|   |            /:::/\:::\    \     
    /:::/|::|   |           /:::/__\:::\    \    
   /:::/ |::|   |           \:::\   \:::\    \   
  /:::/  |::|___|______   ___\:::\   \:::\    \  
 /:::/   |::::::::\    \ /\   \:::\   \:::\    \ 
/:::/    |:::::::::\____/::\   \:::\   \:::\____\
\::/    / ~~~~~/:::/    \:::\   \:::\   \::/    /
 \/____/      /:::/    / \:::\   \:::\   \/____/ 
             /:::/    /   \:::\   \:::\    \     
            /:::/    /     \:::\   \:::\____\    
           /:::/    /       \:::\  /:::/    /    
          /:::/    /         \:::\/:::/    /     
         /:::/    /           \::::::/    /      
        /:::/    /             \::::/    /       
        \::/    /               \::/    /        
         \/____/                 \/____/         
          
         /=============================\
        ||[Virtual Reality Integration]||
         \=============================/                                     
""")


print("""================================================
That is the training session Of MarcoSmiles Virtual Reality Integration
You are training a Double Deep Q Network in order to recognize hand positions
""")

print("EPOCHS ", epocs)
print("Total features in the dataset ", episodes)

notes = np.unique(labels)
training_notes=""
for note in notes:
    training_notes+=str(note) + "|--|"
print("Notes in the dataset (labels)")
print(training_notes[:-4])
print("the training will start in ")
for i in range(5, 0, -1):
        sys.stdout.write(f"\r{i}...")
        sys.stdout.flush()
        time.sleep(1)



for e in range(epocs):

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward


            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                #print("Episode " + str(episode) + ": " + str(episode_reward))
                #print("reward "+ str(episode_reward))
                break

            state = next_state

        print(f"EPOCHS {e+1}/{epocs}   Episode: {episode + 1}, Total Reward: {episode_reward}",end='\r')

    #print(episode_rewards)
    epoch_rewards.append(episode_rewards)
#print(epoch_rewards)
#plot_rewards_line(epoch_rewards)

todaydate=datetime.now()
date_format = "%Y_%m_%d_%H_%M_%S"
model_name="agent_e"+str(epocs)+"_"+todaydate.strftime(date_format)

model_name="HGMSD_24_"+str(epocs)+"ep_DDQN"
#Saving model and optimizer
torch.save(agent.model.state_dict(), model_name+'_model.pth')
torch.save(agent.optimizer.state_dict(), model_name+'_optimizer.pth')
# Test dell'agente addestrato
#state = env.reset()
#total_reward = 0
env.close()
