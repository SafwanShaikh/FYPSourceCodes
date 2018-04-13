import numpy as np
import pandas as pd

data = pd.read_csv("DBChain.csv")
ITC = np.array(data['Introduction to Computer Science'])
CP = np.array(data['Computer Programming'])
DS = np.array(data['Data Structures'])
DB = np.array(data['Database Systems'])
testDataLength = 640
totalDataLength = 1017
countDS_A, countDS_B, countDS_C, countDS_D = (0, 0, 0, 0)
countDB_A, countDB_B, countDB_C, countDB_D = (0, 0, 0, 0)
for i in range(testDataLength):
    if DS[i] == 1.0 or DS[i] == 1.33:
        countDS_D += 1
    elif DS[i] == 1.67 or DS[i] == 2 or DS[i] == 2.33:
        countDS_C += 1
    elif DS[i] == 2.67 or DS[i] == 3 or DS[i] == 3.33:
        countDS_B += 1
    elif DS[i] == 3.67 or DS[i] == 4:
        countDS_A += 1
    i = i+1

for i in range(testDataLength):
    if DB[i] == 1.0 or DB[i] == 1.33:
        countDB_D += 1
    elif DB[i] == 1.67 or DB[i] == 2 or DB[i] == 2.33:
        countDB_C += 1
    elif DB[i] == 2.67 or DB[i] == 3 or DB[i] == 3.33:
        countDB_B += 1
    elif DB[i] == 3.67 or DB[i] == 4:
        countDB_A += 1
    i = i+1


DSaDBaProb = (countDS_A + countDB_A) / testDataLength
DSaDBbProb = (countDS_A + countDB_B) / testDataLength
DSaDBcProb = (countDS_A + countDB_C) / testDataLength
DSaDBdProb = (countDS_A + countDB_D) / testDataLength

DSbDBaProb = (countDS_B + countDB_A) / testDataLength
DSbDBbProb = (countDS_B + countDB_B) / testDataLength
DSbDBcProb = (countDS_B + countDB_C) / testDataLength
DSbDBdProb = (countDS_B + countDB_D) / testDataLength

DScDBaProb = (countDS_C + countDB_A) / testDataLength
DScDBbProb = (countDS_C + countDB_B) / testDataLength
DScDBcProb = (countDS_C + countDB_C) / testDataLength
DScDBdProb = (countDS_C + countDB_D) / testDataLength

DSdDBaProb = (countDS_D + countDB_A) / testDataLength
DSdDBbProb = (countDS_D + countDB_B) / testDataLength
DSdDBcProb = (countDS_D + countDB_C) / testDataLength
DSdDBdProb = (countDS_D + countDB_D) / testDataLength

totalProb = (testDataLength / totalDataLength)

print("DS a DB a: ", DSaDBaProb, "\nDS a DB b: ", DSaDBbProb, "\nDS a DB c: ", DSaDBcProb, "\nDS a DB d: ", DSaDBdProb, "\nDS b DB a: ", DSbDBaProb, "\nDS b DB b: ", DSbDBbProb, "\nDS b DB c: ", DSbDBcProb, "\nDS b DB d: ", DSbDBdProb, "\nDS c DB a: ", DScDBaProb, "\nDS c DB b: ", DScDBbProb, "\nDS c DB c: ", DScDBcProb, "\nDS c DB d: ", DScDBdProb, "\nDS d DB a: ", DSdDBaProb, "\nDS d DB b: ", DSdDBbProb, "\nDS d DB c: ", DSdDBcProb, "\nDS d DB d: ", DSdDBdProb)

passProb16 = (DSaDBaProb+DSaDBbProb+DSaDBcProb+DSaDBdProb+DSbDBaProb+DSbDBbProb+DSbDBcProb+DSbDBdProb+DScDBaProb+DScDBbProb+DScDBcProb+DScDBdProb+DSdDBaProb+DSdDBbProb+DSdDBcProb+DSdDBdProb)/totalProb

print("\nPass Probability(16): ", passProb16)


print("\nA probability: ", DSaDBaProb, " \nB probability: ", DSbDBbProb,"\nC probability", DScDBcProb, "\nD probability", DSdDBdProb)

passProb4 = (DSaDBaProb+DSbDBbProb+DScDBcProb+DSdDBdProb)/totalProb

print("\nPass Probability(4)", passProb4)
