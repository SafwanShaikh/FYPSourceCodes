import numpy as np
import pandas as pd

df = pd.read_csv('weightedTBW.csv')
sum = np.array(df['SUM1'])
DB = np.array(df['Technical and Business Writing'])
sum = sum.reshape((len(sum), 1))

for i in range(len(DB)):
    if DB[i] == 50:
        DB[i] = 1
    elif DB[i] == 54:
        DB[i] = 1
    elif DB[i] == 58:
        DB[i] = 2
    elif DB[i] == 62:
        DB[i] = 2
    elif DB[i] == 66:
        DB[i] = 2
    elif DB[i] == 70:
        DB[i] = 3
    elif DB[i] == 74:
        DB[i] = 3
    elif DB[i] == 78:
        DB[i] = 3
    elif DB[i] == 82:
        DB[i] = 4
    elif DB[i] == 86:
        DB[i] = 4


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
X = sum
y = DB
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
    #print(accuracy_score(y_test, prediction)*100)
    #print(accuracy_score(y_test, prediction, normalize=False)) #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.

print(np.mean(acc))

#X_train, X_test, y_train, y_test = train_test_split(sum, DB, test_size=0.20)
#X_train = np.array(sum[:228])
#y_train = np.array(DB[:228])
#X_test = np.array(sum[-71:])
#y_test = np.array(DB[-71:])
