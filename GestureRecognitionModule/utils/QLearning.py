import numpy as np
import random
import os
import csv
import pandas as pd
import time



n=-1
class State:
    def __init__(self, left_hand, right_hand):
        self.left_hand = left_hand
        self.right_hand = right_hand


    def toString(self):
        temp =""
        for value in self.left_hand:
            temp = temp+str(value)+","
        for value in self.right_hand:
            temp = temp+str(value)+","
        temp=temp[:-1]
        return temp



class Action:
    def __init__(self, chosen_note):
        self.chosen_note = chosen_note


class QLearningAgent:
    def __init__(self):
        self.number_of_states=0
        self.new_note=-1
        self.features = []
        self.distance_array = []
        self.label_array = []
        self.actual_features = None
        self.note_label = 0
        self.path = ""
        self.filename = "marcosmiles_dataset.csv"
        self.folder_name = "MyDatasets"
        self.default_folder = "DefaultDataset"
        self.selected_dataset = self.default_folder
        self.epochs = 10
        self.a_dataset_path = ""
        self.work_dir="ADD your root path here \AppData\LocalLow\DefaultCompany\MarcosMilesOculusIntegration"
        self.num_bins= 4
        self.item_in_bins = [0] * self.num_bins


        # Q-learning parameters
        self.Q_table = {}
        self.alpha = 0.2  # learning rate
        self.gamma = 0.05  # discount factor
        self.epsilon = 0.3  # exploration-exploitation trade-off

    def start(self):
        # read the dataset to start the train
        self.path_starter = "/".join([self.work_dir, self.folder_name, self.selected_dataset, "open_close_starter_data.csv"])
        self.path = "/".join([self.work_dir,self.folder_name, self.selected_dataset, "datasetname.csv"])
        self.retrieve_dataset_pd()

        for i in range(self.epochs):

            #print(f"EPOCHS {i}",end)
            labels_counter = 0
            for d_array in self.features:
                print(f"EPOCHS {i} /{self.epochs} ---labels_counter {labels_counter}",end='\r')
                self.new_note=-1
                self.note_label = self.label_array[labels_counter]
                #print(d_array,"label ",self.note_label)

                self.train(self.create_train_state(d_array))
                #print("returned true")
                labels_counter += 1
            print("\n -------------------------------------------------------------")
        print(self.number_of_states)

    def create_train_state(self, distance_array):
        l_hand = distance_array[:24]
        r_hand = distance_array[24:]

        return State(l_hand, r_hand)

    def train(self, state):
        current_state = state

        while not self.is_terminal_state():

            chosen_action = self.epsilon_greedy_policy(current_state)
            #print(chosen_action.chosen_note)
            reward = self.take_action(chosen_action, current_state)
            self.update_q_value(current_state, chosen_action, reward)
       

    def epsilon_greedy_policy(self, current_state):
        if random.random() < self.epsilon:

            return Action(self.random_note())
        else:
            return self.get_best_action(current_state)

    def update_q_value(self, current_state, chosen_action, reward):
        
        if current_state.toString() not in self.Q_table:
            self.number_of_states+=1
            self.Q_table[current_state.toString()] = {}

        current_q_value = self.Q_table[current_state.toString()].get(chosen_action, 0)
        max_future_q_value = self.get_max_q_value(current_state)

        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_future_q_value)

        self.Q_table[current_state.toString()][chosen_action.chosen_note] = new_q_value

    def take_action(self, action, state):
        self.new_note = action.chosen_note
        #print("in take action ",self.new_note)
        if self.is_terminal_state():
            reward = 100
        else:
            #punishment = abs(self.new_note - self.note_label) * 10
            #reward = 100 - punishment
            reward=0
        return reward

    def get_best_action(self, state):
        if state.toString() in self.Q_table:
            max_q_value = float("-inf")
            best_action = None

            for chosen_action, q_value in self.Q_table[state.toString()].items():
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = Action(chosen_action)



            return best_action

        # If state is not in Q_table, explore by choosing a random action
        return Action(self.random_note())

    def get_max_q_value(self, state):
        if state in self.Q_table:
            return max(self.Q_table[state.toString()].values(), default=0)

        return 0

    def is_terminal_state(self):
        #print("that is the note ",self.new_note)
        #print("that is the label",self.note_label)
        #time.sleep(0.25)
        if self.note_label == self.new_note:
            #print(f"reached terminal state {self.note_label}")
            return True
        
        return False

    def random_note(self):
        n=random.randint(0, 23)


        return n
   

    def retrieve_dataset_pd(self):
        # print(self.current_step)
        df = pd.read_csv(self.path, header=None)

        df.columns = ["label"] + [f"feature_{i}" for i in range(1, len(df.columns))]

        df_starter = pd.read_csv(self.path_starter, header=None)

        df_starter.columns = ["label"] + [f"feature_{i}" for i in range(1, len(df.columns))]

        min_max_distance_df = pd.DataFrame(columns=['column', 'min', 'max', 'dist'])
        #print(df_starter)
        for column in df_starter:

            if (column != 'label'):
                min_val = df_starter[column].min()

                max_val = df_starter[column].max()
                distance = max_val - min_val
                min_max_distance_df = min_max_distance_df._append({'column': column, 'min': min_val, 'max': max_val, 'dist': distance}, ignore_index=True)

        # min_max_distance_df = min_max_distance_df.sort_values(by='dist', ascending=False)
        print(min_max_distance_df)


        multiplier = 10000


        min_max_distance_df.to_csv(
            r'YOUR ROOT PATH HERE\AppData\LocalLow\DefaultCompany\MarcosMilesOculusIntegration\MyDatasets\DefaultDataset\min_max_distance.csv',
            index=False)

        min_max_distance_df_sorted = min_max_distance_df.sort_values(by='dist', ascending=False)
        #print(min_max_distance_df)
        #to_ignore = min_max_distance_df_sorted.drop(min_max_distance_df_sorted.index[:20])
        print(min_max_distance_df_sorted)

        to_save = [3,8,13,18,23,27,32,37,42,47]
        to_ignore = min_max_distance_df.drop(to_save)
        print("this is to ignore")
        print(to_ignore)
        min_array= min_max_distance_df['min'].values * multiplier
        dist_array = min_max_distance_df['dist'].values * multiplier
        print("min and dist")
        print(min_array)
        print(dist_array)

      


        print(df)

        for index, row in df.iterrows():
            print("row ",index,end='\r')
            #remove the label
            row = row[1:]
            examinating_feature=1
            for column,min,dist in zip(row,min_array,dist_array):
                
                bin_width = dist / self.num_bins

                self.item_in_bins = [0] * self.num_bins
                # calculate the bin limits (min * index * width)

                

                bin_limits = [min + ((i+1) * bin_width) for i in range(self.num_bins)]
                #print("THAT IS BIN LIMITS")

                #print(bin_limits[:-1])
                #time.sleep(100)


                for bin_index, limit in enumerate(bin_limits[:-1]):
                    column= column*multiplier
                    if column < limit:
                        self.item_in_bins[bin_index] += 1


                        df.iloc[index,examinating_feature] = bin_index
                        break
                    else:

                        self.item_in_bins[self.num_bins - 1] += 1
                        df.iloc[index, examinating_feature] = self.num_bins - 1
                #print('feature_' + str(examinating_feature - 1))

                to_check = 'feature_'+str((examinating_feature))
                #IF FALSE IGNORE THE FEATURE SELECTION
                if to_check in to_ignore['column'].values and True:
                    #print(" ",to_check, "  ignored")
                    df.iloc[index, examinating_feature]=0
                    self.item_in_bins[0]+=1
                #else:
                    #print(to_check, " ---------- ", column)
                    #print(bin_limits)
                    #print(min)
                    
                #print(df)
                examinating_feature+=1
                #time.sleep(2)
              
                

            #print(df)

        count = 0
        for item in self.item_in_bins:
            print("Numbers of ", count, " = ", item)
            count += 1

        print("\n discretizartion phase finished")
        print(df)


        # Estrai le etichette (y) e le features (X) come array NumPy
        y = df["label"].values
        X = df.drop("label", axis=1).values
        # print(type(y))

        self.label_array = y
        self.features = X;

    def save_qtable_to_csv(self, file_path):
        total_states = len(self.Q_table)
        total_row=0
        #print(self.Q_table)
        for state, actions in self.Q_table.items():
            print(state)
            for action, q_value in actions.items():
                #print(action)
                #print(q_value)
                total_row+=1
        print("TOTAL STATES =",total_states)
        print("TOTAL ROW = ",total_row)
        current_state = 0

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Scrivi l'intestazione
            writer.writerow(['State', 'Action', 'QValue'])
            total_reward_actions=0
            for state, actions in self.Q_table.items():
                print(state)
                for action, q_val in actions.items():
                    percentuale_avanzamento = (current_state / total_row) * 100
                    if(q_val>0):
                        total_reward_actions += 1
                        current_state += 1
                        percentuale_avanzamento = (current_state / total_row) * 100
                        writer.writerow([state, action, q_val])
                    print(f"Salvataggio: {percentuale_avanzamento:.2f}% Complete" ,"saving item ",current_state," --- total row with reward action", total_reward_actions, end="\r")

                    #print(f"Salvataggio: {percentuale_avanzamento:.2f}% Complete" ,"saving item ",current_state," --- total row with reward action", total_reward_actions, end="\r")
                    

                    


        print("\nSalvataggio completato!")

    def save_qtable(self):
        file_path = "/".join([self.work_dir,self.folder_name, self.selected_dataset, "QTable.csv"])
        if os.path.exists(file_path):
            os.remove(file_path)
        self.save_qtable_to_csv(file_path)

# Create an instance of QLearningAgent
q_agent = QLearningAgent()

# Start training
q_agent.start()
print("training completed")
q_agent.save_qtable()
print("qtable saved")
