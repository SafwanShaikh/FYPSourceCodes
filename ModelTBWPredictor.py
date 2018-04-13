import numpy as np
import pandas as pd

df = pd.read_csv('weightedTBW.csv')
ITC = np.array(df['ENG1'])
CP = np.array(df['ENG2'])
DB = np.array(df['Technical and Business Writing'])
AllGPAs = []
for i in range(len(ITC)):
    AllGPAs.append(ITC[i])
    AllGPAs.append(CP[i])
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
GPAs = GPAs.reshape(int(len(AllGPAs)/2), 2)

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
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(GPAs, DB, test_size=0.20)
#X_train = np.array(GPAs[:220])
#y_train = np.array(DB[:220])
clf.fit(X_train, y_train)
#X_test = np.array(GPAs[-75:])
#y_test = np.array(DB[-75:])
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
