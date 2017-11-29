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
GPAs = np.array(AllGPAs)

for i in range(len(GPAs)):
    if GPAs[i] == 1.33:
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
    if DB[i] == 1.33:
        DB[i] = 2
    elif DB[i] == 1.67:
        DB[i] = 2
    elif DB[i] == 2.0:
        DB[i] = 2
    elif DB[i] == 2.33:
        DB[i] = 3
    elif DB[i] == 2.67:
        DB[i] = 3
    elif DB[i] == 3.0:
        DB[i] = 3
    elif DB[i] == 3.33:
        DB[i] = 4
    elif DB[i] == 3.67:
        DB[i] = 4
    elif DB[i] == 4.0:
        DB[i] = 4

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
X_trainCP = np.array(GPAs[-220:])
Y_trainCP = np.array(DB[-220:])
clf.fit(X_trainCP, Y_trainCP)
X_test = np.array(GPAs[:79])
Y_test = np.array(DB[:79])
Y_test = Y_test.ravel()
prediction = clf.predict(X_test)
prediction = prediction.ravel()
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction))
print(accuracy_score(Y_test, prediction, normalize=False)) #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
