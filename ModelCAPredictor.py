import numpy as np
import pandas as pd

df = pd.read_csv('CAChain.csv')
Phy = np.array(df['Physics'])
DLD = np.array(df['Digital Logic Design'])
COAL = np.array(df['Comp. Organization & Assembly Lang.'])
OS = np.array(df['Operating Systems'])
CA = np.array(df['Computer Architecture'])

AllGPAs = []
for i in range(len(Phy)):
    AllGPAs.append(Phy[i])
    AllGPAs.append(DLD[i])
    AllGPAs.append(COAL[i])
    AllGPAs.append(OS[i])

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
GPAs = GPAs.reshape(int(len(AllGPAs)/4), 4)

for i in range(len(CA)):
    if CA[i] == 1.33:
        CA[i] = 2
    elif CA[i] == 1.67:
        CA[i] = 2
    elif CA[i] == 2.0:
        CA[i] = 2
    elif CA[i] == 2.33:
        CA[i] = 3
    elif CA[i] == 2.67:
        CA[i] = 3
    elif CA[i] == 3.0:
        CA[i] = 3
    elif CA[i] == 3.33:
        CA[i] = 4
    elif CA[i] == 3.67:
        CA[i] = 4
    elif CA[i] == 4.0:
        CA[i] = 4

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
X_trainCP = np.array(GPAs[:220])
Y_trainCP = np.array(CA[:220])
clf.fit(X_trainCP, Y_trainCP)
X_test = np.array(GPAs[-73:])
Y_test = np.array(CA[-73:])
Y_test = Y_test.ravel()
prediction = clf.predict(X_test)
prediction = prediction.ravel()
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction))
print(accuracy_score(Y_test, prediction, normalize=False)) #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.

#import matplotlib.pyplot as plt
#import pylab

#plt.scatter(CP, DB)

#plt.show()

#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.pairplot(df, hue="Database Systems") #Variable in data to map plot aspects to different colors.

#plt.show()
#print(df)
