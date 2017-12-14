import numpy as np
import pandas as pd

df = pd.read_csv('weightedCA.csv')
sum = np.array(df['SUM5'])
CA = np.array(df['Computer Architecture'])
sum = sum.reshape((len(sum), 1))

for i in range(len(CA)):
    if CA[i] == 50:
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
X = sum
y = CA
kf = KFold(n_splits=30)
acc = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_test = y_test.ravel()
    prediction = clf.predict(X_test)
    prediction = prediction.ravel()
    from sklearn.metrics import accuracy_score
    acc.append(accuracy_score(y_test, prediction)*100)

print(np.mean(acc))
