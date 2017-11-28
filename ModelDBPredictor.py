import numpy as np
import pandas as pd

df = pd.read_csv('DBchain.csv')
ITC = np.array(df['Introduction to Computer Science'])
CP = np.array(df['Computer Programming'])
DS = np.array(df['Data Structures'])
DB = np.array(df['Database Systems'])
AllGPAs = []
for i in range(len(ITC)):
    AllGPAs.append(ITC[i])
    AllGPAs.append(CP[i])
    AllGPAs.append(DS[i])
    AllGPAs.append(DB[i])
GPAs = np.array(AllGPAs)
