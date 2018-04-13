import numpy as np
import pandas as pd

df = pd.read_csv('DBChain.csv')
ITC = np.array(df['Introduction to Computer Science'])
CP = np.array(df['Computer Programming'])
DS = np.array(df['Data Structures'])
DB = np.array(df['Database Systems'])
AllGPAs = []
for i in range(len(ITC)):
    AllGPAs.append(ITC[i])
    AllGPAs.append(CP[i])
    AllGPAs.append(DS[i])
GPAs = np.array(AllGPAs)

for i in range(len(GPAs)):
    if GPAs[i] == 1.00:
        GPAs[i] = 1
    elif GPAs[i] == 1.33:
        GPAs[i] = 2
    elif GPAs[i] == 1.67:
        GPAs[i] = 3
    elif GPAs[i] == 2.0:
        GPAs[i] = 4
    elif GPAs[i] == 2.33:
        GPAs[i] = 5
    elif GPAs[i] == 2.67:
        GPAs[i] = 6
    elif GPAs[i] == 3.0:
        GPAs[i] = 7
    elif GPAs[i] == 3.33:
        GPAs[i] = 8
    elif GPAs[i] == 3.67:
        GPAs[i] = 9
    elif GPAs[i] == 4.0:
        GPAs[i] = 10
GPAs = GPAs.reshape(int(len(AllGPAs)/3), 3)

for i in range(len(DB)):
    if DB[i] == 1.0:
        DB[i] = 1
    elif DB[i] == 1.33:
        DB[i] = 1
    elif DB[i] == 1.67:
        DB[i] = 2
    elif DB[i] == 2.0:
        DB[i] = 2
    elif DB[i] == 2.33:
        DB[i] = 2
    elif DB[i] == 2.67:
        DB[i] = 3
    elif DB[i] == 3.0:
        DB[i] = 3
    elif DB[i] == 3.33:
        DB[i] = 3
    elif DB[i] == 3.67:
        DB[i] = 4
    elif DB[i] == 4.0:
        DB[i] = 4


cor_df = df.drop(['ID'], axis=1)
#print("Before Pre-processing(9 classes):")
print(cor_df.corr())
temp_df = cor_df
di_other = {1.00: "1", 1.33: "2", 1.67: "3", 2.00: "4", 2.33: "5", 2.67: "6", 3.00: "7", 3.33: "8", 3.67: "9", 4.0: "10"}
di_db = {1.00: "1", 1.33: "1", 1.67: "2", 2.00: "2", 2.33: "2", 2.67: "3", 3.00: "3", 3.33: "3", 3.67: "4", 4.0: "4"}
temp_df = temp_df.replace({"Introduction to Computer Science": di_other, "Computer Programming": di_other, "Data Structures": di_other, "Database Systems": di_db})
print("After Pre-processing(4 classes):")
print(temp_df.corr())

"""
import matplotlib.pyplot as plt
plt.scatter(ITC, DB)
plt.xlabel('ITC')
plt.ylabel('DB')
plt.show()
"""
