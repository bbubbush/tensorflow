import tensorflow as tf
import numpy as np
# 데이터 파싱부터 막힘 ㅎㅎ;;

test = np.loadtxt('gender_submission.csv', dtype={'names': ('PassengerId', 'Survived'), 'formats': ('S4', 'f4')}, delimiter=',', skiprows=True)
print(type(test), type(test[0][0]), type(test[0][1]))

dtype = {"names": ('PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'),
"format": ('S10', 'S10', 'S1', 'S100', 'S10', 'f12', 'S10', 'S10', 'S10', 'f400', 'S10', 'S10')
}
xy = np.loadtxt('train.csv', delimiter=',', skiprows=True, dtype = dtype)




