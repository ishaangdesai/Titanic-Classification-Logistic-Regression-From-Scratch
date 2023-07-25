import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def zScoreNormalization(dataframe):
    for column in dataframe.columns:
        mean = dataframe[column].mean() 
        std = dataframe[column].std() 
        dataframe[column] = (dataframe[column] - mean) / std 
    return dataframe
df = pd.read_csv("titanic/train.csv")
sex = df["Sex"].values.T
#making catagorical feature useful by converting to binary values for each category
is_male = [1 if i=="male" else 0 for i in sex]
is_female = [0 if i=="male" else 1 for i in sex]
#adding new features to dataframe
df["isMale"]=is_male
df["isFemale"]=is_female

#taking unwanted values when converting to x_train
x_train = df.drop(['PassengerId', 'Survived', 'Name', 'Cabin', 'Embarked', 'Sex', 'Ticket'], axis=1)
y_train = df.Survived.values
#since some columns are empty, calculating average age and populating empty NaN values with the average
average = sum(x_train.loc[x_train['Age']>0]["Age"].values.T)/len(x_train.loc[x_train['Age']>0]["Age"].values.T)
x_train = x_train.fillna(average)

x_train = np.array(zScoreNormalization(x_train))
#only using class, isMale, isFemale, age, siblingSpouse, parentChild, fare
#psuedo random values for initial values
np.random.seed(0)
w = np.random.random(7)/10
b = 0.01

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
print(x_train.shape, y_train.shape)
lr.fit(x_train,y_train)
print("Test Accuracy {}".format(lr.score(x_train,y_train)))