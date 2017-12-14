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

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()

X_train = np.array(GPAs[:228])
Y_train = np.array(DB[:228])
clf.fit(X_train, Y_train)
X_test = np.array(GPAs[-71:])
Y_test = np.array(DB[-71:])
Y_test = Y_test.ravel()
prediction = clf.predict(X_test)
prediction = prediction.ravel()
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, prediction)*100)
print(accuracy_score(Y_test, prediction, normalize=False)) #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.

#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.pairplot(df, hue="Database Systems") #Variable in data to map plot aspects to different colors.
#plt.show()
