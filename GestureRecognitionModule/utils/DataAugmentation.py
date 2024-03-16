import pandas as pd
import numpy as np
import csv
import os
import random

file_path = "the dataset you want to increment"
file_path_AUGMENTED="the resulted augmented dataset path"

df = pd.read_csv(file_path, header=None)


df.columns = ["label"] + [f"feature_{i}" for i in range(1, len(df.columns))]
print(df)

# Estrai le etichette (y) e le features (X) come array NumPy
y = df["label"].values
X = df.drop("label", axis=1).values

# Ora y è un array NumPy con le etichette, e X è un array NumPy con le features

# Stampa i due array
#print("Array delle etichette (y):")
#print(y)

#print("Array delle features (X):")
#print(X)
if os.path.exists(file_path_AUGMENTED):
    # Elimina il file
    os.remove(file_path_AUGMENTED)


# Aggiungi le nuove righe al file CSV una alla volta
with open(file_path_AUGMENTED, mode='a', newline='') as file:
	writer = csv.writer(file)
	counter = 0
	for row in X:
		
		for i in range (500):

			tempArr=np.array([])
			#print(str(y[counter]))
			tempArr=np.append(tempArr,str(y[counter]))
			print("GENERATING A ROW")
			for value in row:
				if(i!=0):

					value = float(format(value,'.3f'))
					int_random_val=random.randint(-2,3)

					variation = int_random_val * 0.001;


					#=======percentage variation

					# percentage_variation = np.random.uniform(0, 0.08)
					# variation = value * percentage_variation
					# print("variation",variation)
					# Aggiungi la variazione alla tua feature esistente
					#action = np.random.choice([-1, 0, 1])
					# print("augmented oF",action)
					# finalvalue = value + (variation * action)

					#==========================
					finalvalue=value + variation


					print("initial ",value," added ",variation,"----",finalvalue)

					#tempArr=np.append(tempArr,round(finalvalue,3))
					tempArr = np.append(tempArr, format(finalvalue,'.3f'))

					

				else:
					tempArr=np.append(tempArr,value)

			print(tempArr)
			writer.writerow(tempArr)
		counter = counter + 1
	print("dataset augmented created");
	#print(X)
	
