import numpy as np
import pandas as pd

df = pd.read_csv('weightedSem1to2.csv')
ITC = np.array(df['Introduction to Computer Science'])
CAL1 = np.array(df['Calculus - I'])
ENG = np.array(df['English Language'])
PST = np.array(df['Pakistan Studies'])
PHY = np.array(df['Physics'])
CP = np.array(df['Computer Programming'])

AllGPAs = []
for i in range(len(ITC)):
    AllGPAs.append(ITC[i])
    AllGPAs.append(CAL1[i])
    AllGPAs.append(ENG[i])
    AllGPAs.append(PST[i])
    AllGPAs.append(PHY[i])

GPAs = np.array(AllGPAs)
GPAs = GPAs.reshape(int(len(AllGPAs)/5), 5)

for i in range(len(CP)):
    if CP[i] == 0:
        CP[i] = 0
    elif CP[i] == 1.0:
        CP[i] = 1
    elif CP[i] == 1.33:
        CP[i] = 1
    elif CP[i] == 1.67:
        CP[i] = 2
    elif CP[i] == 2.0:
        CP[i] = 2
    elif CP[i] == 2.33:
        CP[i] = 2
    elif CP[i] == 2.67:
        CP[i] = 3
    elif CP[i] == 3.0:
        CP[i] = 3
    elif CP[i] == 3.33:
        CP[i] = 3
    elif CP[i] == 3.67:
        CP[i] = 4
    elif CP[i] == 4.0:
        CP[i] = 4

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(GPAs, CP, test_size=0.25)
#X_train = np.array(GPAs[:122])
#y_train = np.array(CP[:122])
clf.fit(X_train, y_train)
#X_test = np.array(GPAs[-54:])
#y_test = np.array(CP[-54:])
y_test = y_test.ravel()
prediction = clf.predict(X_test)
prediction = prediction.ravel()
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction)*100)
print(accuracy_score(y_test, prediction, normalize=False)) #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.

#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.pairplot(df, hue="Database Systems") #Variable in data to map plot aspects to different colors.
#plt.show()
