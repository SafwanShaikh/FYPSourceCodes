import numpy as np
import pandas as pd

data = pd.read_csv("DBChain.csv")
ITC = np.array(data['Introduction to Computer Science'])
CP = np.array(data['Computer Programming'])
DS = np.array(data['Data Structures'])
DB = np.array(data['Database Systems'])
AllGPAs = []
for i in range(len(ITC)):
    AllGPAs.append(ITC[i])
    AllGPAs.append(CP[i])
    AllGPAs.append(DS[i])
GPAs = np.array(AllGPAs)
for i in range(len(GPAs)):
    if GPAs[i] == 1.00:
        GPAs[i] = 0
    elif GPAs[i] == 1.33:
        GPAs[i] = 0
    elif GPAs[i] == 1.67:
        GPAs[i] = 1
    elif GPAs[i] == 2.0:
        GPAs[i] = 1
    elif GPAs[i] == 2.33:
        GPAs[i] = 1
    elif GPAs[i] == 2.67:
        GPAs[i] = 2
    elif GPAs[i] == 3.0:
        GPAs[i] = 2
    elif GPAs[i] == 3.33:
        GPAs[i] = 2
    elif GPAs[i] == 3.67:
        GPAs[i] = 3
    elif GPAs[i] == 4.0:
        GPAs[i] = 3

for i in range(len(DB)): #(299,1)
    if DB[i] == 1.0:
        DB[i] = 0
    elif DB[i] == 1.33:
        DB[i] = 0
    elif DB[i] == 1.67:
        DB[i] = 1
    elif DB[i] == 2.0:
        DB[i] = 1
    elif DB[i] == 2.33:
        DB[i] = 1
    elif DB[i] == 2.67:
        DB[i] = 2
    elif DB[i] == 3.0:
        DB[i] = 2
    elif DB[i] == 3.33:
        DB[i] = 2
    elif DB[i] == 3.67:
        DB[i] = 3
    elif DB[i] == 4.0:
        DB[i] = 3

GPAs = GPAs.reshape(int(len(AllGPAs)/3), 3)     #(739, 3)
predictors = GPAs[:589]  # predictor
target = DB[:589]    # target
# 590 in sheet
testDataX = GPAs[-150:]
testDataY = DB[-150:]

"""
from keras.utils import to_categorical
predictors = to_categorical(predictors)
target = to_categorical(target)
"""

n_cols = predictors.shape[1]        # n_cols = 3

import keras
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(3, activation='relu', input_shape=(n_cols, )))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(predictors, target, batch_size=100, epochs=100)


yPredict = model.predict(testDataX)
yPredictL = []

for i in range(len(yPredict)):
    maxValue = max(yPredict[i])
    temp = np.array(yPredict[i])
    for j in range(len(temp)):
        if(maxValue == temp[j]):
            yPredictL.append(j)

from sklearn.metrics import accuracy_score
print(accuracy_score(testDataY, yPredictL))
