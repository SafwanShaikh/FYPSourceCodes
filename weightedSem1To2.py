import numpy as np
import pandas as pd

df = pd.read_csv('weightedSem1to2.csv')
sum = np.array(df['SUM1'])
CA = np.array(df['Computer Programming'])
sum = sum.reshape((len(sum), 1))

for i in range(len(CA)):
    if CA[i] == 0:
        CA[i] = 0
    elif CA[i] == 50:
        CA[i] = 1
    elif CA[i] == 54:
        CA[i] = 1
    elif CA[i] == 58:
        CA[i] = 2
    elif CA[i] == 62:
        CA[i] = 2
    elif CA[i] == 66:
        CA[i] = 2
    elif CA[i] == 70:
        CA[i] = 3
    elif CA[i] == 74:
        CA[i] = 3
    elif CA[i] == 78:
        CA[i] = 3
    elif CA[i] == 82:
        CA[i] = 4
    elif CA[i] == 86:
        CA[i] = 4


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(sum, CA, test_size=0.30)
#X_train = np.array(sum[:122])
#y_train = np.array(CA[:122])
#X_test = np.array(sum[-54:])
#y_test = np.array(CA[-54:])
clf.fit(X_train, y_train)
y_test = y_test.ravel()
prediction = clf.predict(X_test)
prediction = prediction.ravel()
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction)*100)
