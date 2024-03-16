from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os 
import numpy as np

## label -1 = Not recognized

ignore_feature_imp=True
important_features=[3,11,27,2,17,46,45,26,37,22,7,1,47,35,8,18,42,32,23]
global_accuracy=[]
global_precision=[]
global_recall=[]
mean_array_accuracy=[]
mean_array_precision=[]
mean_array_recall=[]
class PoseChecker:
    def __init__(self, features, label):
        self.features = features
        self.label = label

    def tostring(self):
        print(self.label)
        print(self.features)
        print("\n")

    def checkvalidity(self,threshold,featurestocheck):
        #print(self.features)
        #print(featurestocheck)
        errorcounter=0
        #print("type of features")
        #print(type(self.features))
        #print("type of features to check")
        #print(type(featurestocheck))
        f_counter=0

        for fp,ftc,th in zip(self.features,featurestocheck,threshold):
            
            if (f_counter in important_features or ignore_feature_imp==True):

                if (ftc > (fp + th)) or (ftc < (fp - th)):
                    errorcounter+=1
            f_counter+=1
        if (errorcounter == 0):
            return self.label
        return -1

    def get_label(self):
        return self.label





def retrieve_dataset_pd():

    path = r"YOUR PATH HERE\TestDataset24.csv"
    df = pd.read_csv(path, header=None)

    df.columns = ["label"] + [f"feature_{i}" for i in range(1, len(df.columns))]

    # Estrai le etichette (y) e le features (X) come array NumPy
    y = df["label"].values
    X = df.drop("label", axis=1).values


    return X,y;


def retrieve_posechecker():

    path = r"YOUR PATH HERE\SIMSD_24.csv"
    df = pd.read_csv(path, header=None)

    df.columns = ["label"] + [f"feature_{i}" for i in range(1, len(df.columns))]

    # Estrai le etichette (y) e le features (X) come array NumPy
    y = df["label"].values
    X = df.drop("label", axis=1).values
    posecheckers=[]
    # init posecheckers
    for xi, yi in zip(X, y):

        posecheckers.append(PoseChecker(xi,yi))
    #return threszholds based on std
    std_for_column = []
    for item in X.std(axis=0):
        cast_to_float = round(item, 3)
        std_for_column.append(cast_to_float)



    min_dist_columns = np.ptp(X, axis=0)
    print(min_dist_columns)


    return posecheckers,std_for_column,min_dist_columns;



X,Y = retrieve_dataset_pd()

Posecheckers,T_for_fingers,min_dist_columns=retrieve_posechecker()
for p in Posecheckers:
    print(p.get_label())

#print(T_for_fingers)

# Nome del file
result_path = r'YOUR PATH HERE\results-24.txt'


if os.path.exists(result_path):
    os.remove(result_path)

table_path=r'C:YOUR PATH HERE\table_SIMDS-24.txt'



if os.path.exists(table_path):
    os.remove(table_path)






for i in range (20):
    # Init confusion_matrix
    num_notes = 24  # Dal 0 al 23
    activation_matrix = np.zeros((num_notes, num_notes),dtype=int)

    global_predictions = []
    local_prediction=[]

    th_array = np.full(48, (i + 1) / 1000)
    
    total_accuracy = []
    total_precision = []
    total_recall=[]
    with open(result_path, 'a') as file:
        file.write(f"=============== result threshold {(i + 1) / 1000} ===================\n")
       
        for posechecker in Posecheckers:
            local_prediction = []
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            rows_examinated=0
            #print(th_array)
            #print("\n=======================================")
            for f,l in zip(X,Y):

                rows_examinated+=1

                #Posecheckers[0].tostring()
                result=posechecker.checkvalidity(th_array,f)
                local_prediction.append(result)



                #print(f"posechecker{l} ---- item {rows_examinated} of {len(X)}",end='\r')
                #print("result ",result,"------lp ",posechecker.get_label())

                if(l==posechecker.get_label()):
                    if(result == l):
                        TP+=1
                    else:
                        FN+=1
                else:
                    if (result == -1):
                        TN += 1
                    else:
                        FP += 1
            file.write("==========================================================\n")
            file.write(f"Results for POSECHECKER {posechecker.get_label()}\n")
            file.write("==========================================================\n")
            file.write(f"Rows examinated: {rows_examinated} \n\n")


            file.write(f"TP: {TP}\n")
            file.write(f"FP: {FP}\n")
            file.write(f"TN: {TN}\n")
            file.write(f"FN: {FN}\n\n")

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            #file.write(f"ACCURACY: {accuracy:.2f}\n")
            file.write(f"ACCURACY: {accuracy}\n")
            precision = 0
            if((TP + FP)!=0):
                precision = TP / (TP + FP)
            #file.write(f"PRECISION: {precision:.2f}\n")
            recall=TP/(TP+FN)

            file.write(f"PRECISION: {precision}\n")
            file.write(f"RECALL: {recall}\n")
            total_accuracy.append(accuracy)
            total_precision.append(precision)
            total_recall.append(recall)


            file.write(f"SUCCESS PERCENTAGE ON GESTURE RECOGNITION: {TP/1500} \n")

            file.write("==========================================================\n")
            #print(local_prediction)
            # Itera sugli elementi dell'array e modifica direttamente quelli che soddisfano le condizioni
            for k in range(len(local_prediction)):
                if local_prediction[k] == -1:
                    local_prediction[k] = False
                else:
                    local_prediction[k] = True
            #print(local_prediction)
            global_predictions.append(local_prediction)

    #print(len(global_predictions))
    # Calcola la matrice di attivazioni
    for b in range(num_notes):
        for c in range(b + 1, num_notes):

            common_activations = np.logical_and(global_predictions[b], global_predictions[c])
            #print(common_activations)
            activation_matrix[b, c] = np.sum(common_activations)
            activation_matrix[b, c] = activation_matrix[b, c]  # Simmetrica, quindi copia il valore

    # Visualizzazione della Confusion Matrix
    file_name=r"YOUR PATH HERE\TH"+str((i+1)/1000)+".png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(activation_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=list(range(24)),
                yticklabels=list(range(24)))
    plt.title('Activation Matrix')
    plt.xlabel('Posechecker')
    plt.ylabel('Posechecker')
    plt.savefig(file_name)
    #plt.show()





    global_accuracy.append(total_accuracy)
    global_recall.append(total_recall)
    global_precision.append(total_precision)
    
    with open(table_path, 'a') as file:
        file.write(f"==================Treshold {(i + 1) / 1000}======================\n")
        file.write(f"avg acc: {np.mean(total_accuracy)}\n")
        file.write(f"avg prec : {np.mean(total_precision)}\n")
        file.write(f"avg recall : {np.mean(total_recall)}\n")
    
    
    
    print(f"==================Treshold {(i+1)/1000}")
    print("avg acc: ", np.mean(total_accuracy))
    print("avg prec : ",np.mean(total_precision))
    print("avg recall : ", np.mean(total_recall))
    mean_array_accuracy.append(np.mean(total_accuracy))
    mean_array_precision.append(np.mean(total_precision))
    mean_array_recall.append(np.mean(total_recall))



max_ma = np.argmax(mean_array_accuracy)
max_ma_t = mean_array_accuracy[max_ma]
print(f"Max accuracy : {max_ma_t}-- treshold: {(max_ma+1)/1000} ")

max_mr = np.argmax(mean_array_recall)
max_mr_t = mean_array_recall[max_mr]
print(f"Max recall  : {max_mr_t}-- treshold: {(max_mr+1)/1000} ")

max_mp = np.argmax(mean_array_precision)
max_mp_t= mean_array_precision[max_mp]
print(f"Max precision  : {max_mp_t}-- treshold: {(max_mp+1)/1000} " )






'''
print("accuracy=======")
print(global_accuracy)
print("recall=======")
print(global_recall)
print("precision=======")
print(global_precision)




Posechecker_counter=0
for a,r,p in zip(global_accuracy,global_recall,global_precision):

    max_a = np.argmax(a)
    max_a_t = a[max_a]
    print(f"Max accuracy for posechecker {Posechecker_counter} : {max_a_t}-- treshold: {(max_a+1)/1000} ")

    max_r = np.argmax(r)
    max_r_t = r[max_r]
    print(f"Max recall for posechecker {Posechecker_counter} : {max_r_t}-- treshold: {(max_r+1)/1000} ")

    max_p = np.argmax(p)
    max_p_t= p[max_p]
    print(f"Max precision for posechecker {Posechecker_counter} : {max_p_t}-- treshold: {(max_p+1)/1000} " )
    Posechecker_counter+=1
'''




