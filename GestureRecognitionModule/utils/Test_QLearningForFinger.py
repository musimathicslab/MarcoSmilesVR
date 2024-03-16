from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
# State class
class State:
    def __init__(self, left_hand, right_hand):
        self.left_hand = left_hand
        self.right_hand = right_hand



    def toString(self):
        temp = ""
        for value in self.left_hand:
            temp = temp + str(value) + ","
        for value in self.right_hand:
            temp = temp + str(value)+","
        temp=temp[:-1]
        print(temp)
        return temp

    @classmethod
    def from_string(cls, state_str):
        values = state_str.split(',')
        left_hand = list(map(float, values[:24]))

        right_hand = list(map(float, values[24:]))


        return cls(left_hand, right_hand)


    @classmethod
    def create_new_state(features):

        # Split the hand features
        left_hand = features[:24]
        right_hand = features[24:]


        # Create and return the state
        return State(left_hand, right_hand)

# Action Class
class Action:
    def __init__(self, chosen_note):
        self.chosen_note = chosen_note







class QLearningAgent:


    def __init__(self):
        self.Q_table= {}
        self.filename = "marcosmiles_dataset.csv"
        self.folder_name = "MyDatasets"
        self.default_folder = "DefaultDataset"
        self.selected_dataset = self.default_folder
        self.a_dataset_path = ""
        self.work_dir = r"YOUR ROOT PATH HERE\AppData\LocalLow\DefaultCompany\MarcosMilesOculusIntegration"
        self.path = ""
        self.label_array=None
        self.features=None
        self.predict=[]
        self.chosed_randomly=0
        self.chosed_from_experience=0
        self.num_bins=10
        self.item_in_bins = [0] * self.num_bins


    def take_optimal_action(self, current_state):
        #print("current")
        #print(current_state.toString())
        #print("in q table")
        #print(self.Q_table[current_state.toString()])
        
        if current_state.toString() in self.Q_table:
            #if state present chose the action qith max q-value
            best_action = max(self.Q_table[current_state.toString()], key=self.Q_table[current_state.toString()].get)
            print("is in the qtable")
            self.chosed_from_experience+=1
            return best_action
        else:
            self.chosed_randomly+=1
            print("chosed randomly")
            # if state not present in the qtable, chose random
            return self.random_action()


    #return a random number from 0 to 25
    def random_action(self):
        n = Action(random.randint(0, 23))
        return n

    def retrieve_distance_pd(self):
        # print(self.current_step)
        self.path = "/".join([self.work_dir, self.folder_name, self.selected_dataset, "min_max_distance.csv"])
        self.min_max_distance_df = pd.read_csv(self.path)



        self.to_ignore = self.min_max_distance_df.drop(self.min_max_distance_df.index[:20])
        print(self.min_max_distance_df)
        print(self.to_ignore)

    def retrieve_dataset_pd(self):
        # print(self.current_step)
        self.path = "/".join([self.work_dir, self.folder_name, self.selected_dataset, "datasetpath.csv"])
        df = pd.read_csv(self.path, header=None)

        df.columns = ["label"] + [f"feature_{i}" for i in range(1, len(df.columns))]
        # min_max_distance_df = min_max_distance_df.sort_values(by='dist', ascending=False)
        min_max_distance_df=self.min_max_distance_df
        # EveryIndex from the left hand (0-4) and right hand (5-9) each one representing a finger
        features_finger_range = [0] * 10
        multiplier = 10000
        # print(min_max_distance_df['dist'].iloc[44:48])
        features_finger_range[0] = min_max_distance_df['dist'].iloc[0:4].sum() * multiplier
        features_finger_range[1] = min_max_distance_df['dist'].iloc[4:9].sum() * multiplier
        features_finger_range[2] = min_max_distance_df['dist'].iloc[9:14].sum() * multiplier
        features_finger_range[3] = min_max_distance_df['dist'].iloc[14:19].sum() * multiplier
        features_finger_range[4] = min_max_distance_df['dist'].iloc[19:24].sum() * multiplier
        # right
        features_finger_range[5] = min_max_distance_df['dist'].iloc[24:28].sum() * multiplier
        features_finger_range[6] = min_max_distance_df['dist'].iloc[28:33].sum() * multiplier
        features_finger_range[7] = min_max_distance_df['dist'].iloc[33:38].sum() * multiplier
        features_finger_range[8] = min_max_distance_df['dist'].iloc[38:43].sum() * multiplier
        features_finger_range[9] = min_max_distance_df['dist'].iloc[43:48].sum() * multiplier

        features_finger_min = [0] * 10

        # Left
        features_finger_min[0] = min_max_distance_df['min'].iloc[0:4].sum() * multiplier
        features_finger_min[1] = min_max_distance_df['min'].iloc[4:9].sum() * multiplier
        features_finger_min[2] = min_max_distance_df['min'].iloc[9:14].sum() * multiplier
        features_finger_min[3] = min_max_distance_df['min'].iloc[14:19].sum() * multiplier
        features_finger_min[4] = min_max_distance_df['min'].iloc[19:24].sum() * multiplier

        # Right
        features_finger_min[5] = min_max_distance_df['min'].iloc[24:28].sum() * multiplier
        features_finger_min[6] = min_max_distance_df['min'].iloc[28:33].sum() * multiplier
        features_finger_min[7] = min_max_distance_df['min'].iloc[33:38].sum() * multiplier
        features_finger_min[8] = min_max_distance_df['min'].iloc[38:43].sum() * multiplier
        features_finger_min[9] = min_max_distance_df['min'].iloc[43:48].sum() * multiplier

        min_max_distance_df.to_csv(
            r'YOUR ROOT PATH HERE\AppData\LocalLow\DefaultCompany\MarcosMilesOculusIntegration\MyDatasets\DefaultDataset\min_max_distance.csv',
            index=False)

        min_max_distance_df = min_max_distance_df.sort_values(by='dist', ascending=False)
        # print(min_max_distance_df)
        to_ignore = min_max_distance_df.drop(min_max_distance_df.index[:20])
        # print(to_ignore)

        print("FEATURE_FINGER_RANGE ----  MIN")
        print(features_finger_range)
        print(features_finger_min)

        finger_indexes = [(0, 4), (4, 9), (9, 14), (14, 19), (19, 24), (24, 28), (28, 33), (33, 38), (38, 43), (43, 48)]

        print(df)

        for index, row in df.iterrows():
            print(f"discretization state {index}", end="'\r")
            discretization_fingers = [0] * 10
            finger_counter = 0
            # print(row)
            for start, end in finger_indexes:
                # print("========================")
                # print(row[1:].iloc[start:end].values)
                discretization_fingers[finger_counter] = (row[1:].iloc[start:end].values.sum()) * multiplier
                # print(discretization_fingers[finger_counter])
                # print("========================")
                finger_counter += 1
            counter_change = 0
            # print("that is discretization finger")
            # print(discretization_fingers)
            for to_convert, ref, min in zip(discretization_fingers, features_finger_range, features_finger_min):
                # print(ref)

                bin_width = ref / self.num_bins

                self.item_in_bins = [0] * self.num_bins
                # calculate the bin limits (min * index * width)
                bin_limits = [min + ((i + 1) * bin_width) for i in range(self.num_bins)]
                # print("THAT IS BIN LIMITS")
                # print(bin_limits)
                # print(bin_limits[:-1])
                # time.sleep(100)

                for bin_index, limit in enumerate(bin_limits[:-1]):
                    # print("to convert", to_convert)
                    # print("ref ", ref)
                    # print(limit)
                    if to_convert < limit:
                        self.item_in_bins[bin_index] += 1
                        start_index, end_index = finger_indexes[counter_change]
                        start_index += 1
                        end_index += 1

                        df.iloc[index, start_index:end_index] = bin_index
                        break
                    else:

                        start_index, end_index = finger_indexes[counter_change]
                        start_index += 1
                        end_index += 1
                        df.iloc[index, start_index:end_index] = self.num_bins - 1

                counter_change += 1
                # time.sleep(2)

        count = 0
        for item in self.item_in_bins:
            print("Numbers of ", count, " = ", item)
            count += 1

        print("\n discretization phase finished")
        print(df)

        # Estrai le etichette (y) e le features (X) come array NumPy
        y = df["label"].values
        X = df.drop("label", axis=1).values

        self.label_array = y
        self.features = X;

    def load_qtable_from_csv(self, file_path):

        with open(file_path,'r') as file:
            num_rows = sum(1 for row in file)



        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Ignore header
            for row in tqdm(reader, total=num_rows):


                state_str, action, q_value = row
                state = State.from_string(state_str)
                #print(state.toString())
                action = Action(int(action))
                #print(action.chosen_note)
                q_value = float(q_value)
                #print(q_value)
                #time.sleep(0.5)
                
                if state not in self.Q_table:
                    #print(state)
                    self.Q_table[state.toString()] = {}



                self.Q_table[state.toString()][action] = q_value

        print(self.Q_table)


    #def epsilon_greedy_policy(self, current_state):
        # Implementa la tua logica per la politica epsilon-greedy qui
        #pass

    def start_testing(self):
        #print(self.Q_table)
        #time.sleep(200)
        for row in self.features:


            # Create the state
            new_state = State(row[:24], row[24:])
            #print(new_state.toString())
            predicted_note=self.take_optimal_action(new_state)
            print(predicted_note.chosen_note)
            self.predict.append(predicted_note.chosen_note)



        print("CHOSED RANDOMLY: ",self.chosed_randomly)
        print("CHOSED FROM EXPERIENCE: ",self.chosed_from_experience)

        #Accuracy
        accuracy = accuracy_score(self.label_array, self.predict)
        print(f'Accuracy: {accuracy}')

        # Precision
        precision = precision_score(self.label_array, self.predict, average='macro')
        print(f'Precision: {precision}')

        # Recall
        recall = recall_score(self.label_array, self.predict, average='micro')
        print(f'Recall: {recall}')

        # F1 score
        f1 = f1_score(self.label_array, self.predict, average='micro')
        print(f'F1 Score: {f1}')

        # Confusion matrix
        conf_matrix = confusion_matrix(self.label_array, self.predict)
        print(f'Confusion Matrix:\n{conf_matrix}')

        # Visualizzazione della Confusion Matrix
        plt.figure(figsize=(16, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(24)), yticklabels=list(range(24)))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        


q_agent = QLearningAgent()
q_agent.retrieve_distance_pd()
q_agent.load_qtable_from_csv(r"YOUR ROOT PATH HERE\AppData\LocalLow\DefaultCompany\MarcosMilesOculusIntegration\MyDatasets\DefaultDataset\QTable.csv")
print(q_agent.Q_table)
q_agent.retrieve_dataset_pd()
q_agent.start_testing()

