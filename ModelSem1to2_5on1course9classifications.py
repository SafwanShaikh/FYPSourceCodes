import pandas as pd
import numpy as np

df = pd.read_csv('CSupdatedSemesterAlloted.csv')
students = pd.DataFrame(df[['Roll No', 'Program', 'Course', 'Course Title', 'Credit Hrs.', 'GP', 'Grade', 'Sem.', 'Num of Semester', 'Batch']])
students = students.dropna()
firstSemesterStudent = students.where(students['Num of Semester'].astype(np.int64) == 1).dropna()
firstSemesterStudent = firstSemesterStudent[['Roll No', 'Course Title', 'Grade', 'GP', 'Batch']]
#print(set(firstSemesterStudent['Course Title']))       # print distinct courses
firstSemesterStudentFilter = firstSemesterStudent[firstSemesterStudent['Course Title'].isin(['Pakistan Studies', 'English Language', 'Physics - I', 'Physics', 'Introduction to Computer Science', 'Calculus - I'])]
#check = firstSemesterStudentFilter[firstSemesterStudentFilter['Roll No'] == '56K-1631']
#print(firstSemesterStudentFilter['Roll No'].unique())
rollNoFirstSemStudentFilter = pd.DataFrame(firstSemesterStudentFilter['Roll No'].unique())
firstSemesterStudentFilter = firstSemesterStudentFilter.set_index(['Roll No'])
finalFirstSemStudent = pd.DataFrame()
for i in range(len(rollNoFirstSemStudentFilter)):
    if len((firstSemesterStudentFilter.ix[rollNoFirstSemStudentFilter.ix[i]])) == 5:
        finalFirstSemStudent = finalFirstSemStudent.append(firstSemesterStudentFilter.ix[rollNoFirstSemStudentFilter.ix[i]])

finalFirstSemStudent.to_csv("Semester1Students.csv", sep=',', encoding='utf-8')
#####################################################################################################\

secondSemesterStudent = students.where(students['Num of Semester'].astype(np.int64) == 2).dropna()
secondSemesterStudent = secondSemesterStudent[['Roll No', 'Course Title', 'GP', 'Grade', 'Batch']]
secondSemesterStudentFilter = secondSemesterStudent[secondSemesterStudent['Course Title'].isin(['Computer Programming', 'Digital Logic Design', 'Physics - II', 'Islamic & Religious Studies', 'English Composition', 'Calculus - II'])]
rollNoSecondSemStudentFilter = pd.DataFrame(secondSemesterStudentFilter['Roll No'].unique())
secondSemesterStudentFilter = secondSemesterStudentFilter.set_index(['Roll No'])
finalSecondSemStudent = pd.DataFrame()
for i in range(len(rollNoSecondSemStudentFilter)):
    if len((secondSemesterStudentFilter.ix[rollNoSecondSemStudentFilter.ix[i]])) == 5:
        finalSecondSemStudent = finalSecondSemStudent.append(secondSemesterStudentFilter.ix[rollNoSecondSemStudentFilter.ix[i]])

finalSecondSemStudent.to_csv("Semester2Students.csv", sep=',', encoding='utf-8')

firstSemesterRollNoList = finalFirstSemStudent.index.unique()
secondSemesterRollNoList = finalSecondSemStudent.index.unique()
print("First Semester students :", len(firstSemesterRollNoList))
print("Second Semester Students :", len(secondSemesterRollNoList))
#diff = set(firstSemesterRollNoList).intersection(secondSemesterRollNoList)
diff = list(set(firstSemesterRollNoList) - set(secondSemesterRollNoList))
intersection = set(firstSemesterRollNoList).intersection(secondSemesterRollNoList)
print("Students that are in first semester but not in second because of fail/withdraw etc :", len(diff))
print("But Difference should be 210 as 388-178 = 210")
print("Common students in Both Semester 1 and 2 :", len(intersection))

studentIn2butNotIn1 = list(set(secondSemesterRollNoList) - set(intersection))
print("Students in second semester but not in 1 because of one course :", studentIn2butNotIn1)


for i in range(len(studentIn2butNotIn1)):
    finalSecondSemStudent.drop([studentIn2butNotIn1[i]], inplace=True)

print('Dropped both students')

for i in range(len(diff)):
    finalFirstSemStudent.drop([diff[i]], inplace=True)

print('Dropped students from sem 1 who fails or withdraw')

finalFirstSemStudent.to_csv("CommonSemester1Students.csv", sep=',', encoding='utf-8')
finalSecondSemStudent.to_csv("CommonSemester2Students.csv", sep=',', encoding='utf-8')

firstSemesterGPAs = np.array(finalFirstSemStudent['GP'])

for i in range(len(firstSemesterGPAs)):
    if firstSemesterGPAs[i] == 1.33:
        firstSemesterGPAs[i] = 2
    elif firstSemesterGPAs[i] == 1.67:
        firstSemesterGPAs[i] = 3
    elif firstSemesterGPAs[i] == 2.0:
        firstSemesterGPAs[i] = 4
    elif firstSemesterGPAs[i] == 2.33:
        firstSemesterGPAs[i] = 5
    elif firstSemesterGPAs[i] == 2.67:
        firstSemesterGPAs[i] = 6
    elif firstSemesterGPAs[i] == 3.0:
        firstSemesterGPAs[i] = 7
    elif firstSemesterGPAs[i] == 3.33:
        firstSemesterGPAs[i] = 8
    elif firstSemesterGPAs[i] == 3.67:
        firstSemesterGPAs[i] = 9
    elif firstSemesterGPAs[i] == 4.0:
        firstSemesterGPAs[i] = 10
firstSemesterGPAs = firstSemesterGPAs.astype(np.int32)
firstSemesterGPAs = firstSemesterGPAs.reshape(int(len(firstSemesterGPAs)/5), 5)

secondSemesterGPAs = np.array(finalSecondSemStudent['GP'])
for i in range(len(secondSemesterGPAs)):
    if secondSemesterGPAs[i] == 1.33:
        secondSemesterGPAs[i] = 2
    elif secondSemesterGPAs[i] == 1.67:
        secondSemesterGPAs[i] = 3
    elif secondSemesterGPAs[i] == 2.0:
        secondSemesterGPAs[i] = 4
    elif secondSemesterGPAs[i] == 2.33:
        secondSemesterGPAs[i] = 5
    elif secondSemesterGPAs[i] == 2.67:
        secondSemesterGPAs[i] = 6
    elif secondSemesterGPAs[i] == 3.0:
        secondSemesterGPAs[i] = 7
    elif secondSemesterGPAs[i] == 3.33:
        secondSemesterGPAs[i] = 8
    elif secondSemesterGPAs[i] == 3.67:
        secondSemesterGPAs[i] = 9
    elif secondSemesterGPAs[i] == 4.0:
        secondSemesterGPAs[i] = 10
secondSemesterGPAs = secondSemesterGPAs.astype(np.int32)

listSecondSemesterCP_GPA = []
j = 0               # For 1st course of second semester j=0, if we train it for second course update j=0 to j=1, for third course j=2 and upto j=4 for fifth course of second semester.
for i in range(int(len(secondSemesterGPAs)/5)):
    listSecondSemesterCP_GPA.append(secondSemesterGPAs[j])
    j = j+5
secondSemesterCP_GPA = np.array(listSecondSemesterCP_GPA)

#secondSemesterGPAs = secondSemesterGPAs.reshape(int(len(secondSemesterGPAs)/5), 5)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
X_trainCP = np.array(firstSemesterGPAs[0:140])
Y_trainCP = np.array(secondSemesterCP_GPA[0:140])
clf.fit(X_trainCP, Y_trainCP)
X_testCP = np.array(firstSemesterGPAs[-37:])
Y_testCP = np.array(secondSemesterCP_GPA[-37:])
Y_testCP = Y_testCP.ravel()
prediction = clf.predict(X_testCP)
prediction = prediction.ravel()
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_testCP, prediction))
print(accuracy_score(Y_testCP, prediction, normalize=False)) #If False, return the number of correctly classified samples. Otherwise, return the fraction of correctly classified samples.
