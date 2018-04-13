import numpy as np
import pandas as pd

df = pd.read_csv('MTchain.csv')
sum = np.array(df['SUM1'])
PB = np.array(df['Probability & Statistics'])
sum = sum.reshape((len(sum), 1))

for i in range(len(PB)):
    if PB[i] == 0:
        PB[i] = 0
    elif PB[i] == 50:
        PB[i] = 1
    elif PB[i] == 54:
        PB[i] = 1
    elif PB[i] == 58:
        PB[i] = 2
    elif PB[i] == 62:
        PB[i] = 2
    elif PB[i] == 66:
        PB[i] = 2
    elif PB[i] == 70:
        PB[i] = 3
    elif PB[i] == 74:
        PB[i] = 3
    elif PB[i] == 78:
        PB[i] = 3
    elif PB[i] == 82:
        PB[i] = 4
    elif PB[i] == 86:
        PB[i] = 4

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
X = sum
y = PB
kf = KFold(n_splits=20)
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
