import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def data_prep(df, y_truth):
    sex = df["Sex"].values.T
    #making catagorical feature useful by converting to binary values for each category
    is_male = [1 if i=="male" else 0 for i in sex]
    is_female = [0 if i=="male" else 1 for i in sex]
    #adding new features to dataframe
    df["isMale"]=is_male
    df["isFemale"]=is_female
    #taking unwanted values when converting to x_train
    #only using class, isMale, isFemale, age, siblingSpouse, parentChild, fare
    
    if y_truth==False:
        y = df.Survived.values
        x = df.drop(['PassengerId', 'Survived', 'Name', 'Cabin', 'Embarked', 'Sex', 'Ticket'], axis=1)
    else:
        x = df.drop(['PassengerId', 'Name', 'Cabin', 'Embarked', 'Sex', 'Ticket'], axis=1)
    #since some columns are empty, calculating average age and populating empty NaN values with the average
    average = sum(x.loc[x['Age']>0]["Age"].values.T)/len(x.loc[x['Age']>0]["Age"].values.T)
    x = x.fillna(average)
    x = z_score_normalization(x)
    x=np.array(x)
    if y_truth==True:
        return x
    return x, y


def z_score_normalization(df):
    for column in df.columns:
        mean = df[column].mean() 
        std = df[column].std() 
        df[column] = (df[column] - mean) / std 
    return df


def sigmoid(z):
    return 1/(1+np.exp(-z))


def forwards_backwards_propagation(x_train, y_train, w, b):
    y_head = sigmoid(np.dot(x_train, w)+b)
    logistic_loss = -y_train * np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = sum(logistic_loss)/x_train.shape[0]

    derivative_w = np.dot(x_train.T, (y_head-y_train))/x_train.shape[0]
    derivative_b = np.sum(y_head-y_train)/x_train.shape[0]
    return cost, derivative_w, derivative_b


def gradient_descent(w, b, x_train, y_train, alpha, iterations):
    #w is a vector with features, b is bias, x_train is input of training dataset, y_train is actual output of training dataset 
    #alpha is learning rate, iterations is number of iterations to perform
    full_cost_list = []

    full_cost_list = []
    for i in range(iterations):
        cost, derivative_w, derivative_b = forwards_backwards_propagation(x_train, y_train, w, b)
        full_cost_list.append(cost)
        w = w - alpha*derivative_w
        b = b - alpha*derivative_b
        if i%10==0:
            print("Iteration %s with cost %s "%(i,cost))
    final_gradients = {'w':derivative_w, 'b':derivative_b}
    final_w_b = {'w':w, 'b':b}
    print(final_w_b)
    return full_cost_list, final_gradients, final_w_b


def predict(w, b, x_test):
    f_wb = sigmoid(np.dot(x_test, w)+b)
    y_prediction = []
    for i in f_wb:
        if i>=0.5:
            y_prediction.append(1)
        else:
            y_prediction.append(0)
    return y_prediction

def logistic_regression(w, b, x_train, y_train,x_test, y_test, learning_rate,  num_iterations):
    cost , gradients ,parameters  = gradient_descent(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["w"],parameters["b"],x_test)
    # Print train/test Errors
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

df = pd.read_csv("titanic/train.csv")
x_train, y_train = data_prep(df, False)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

#psuedo random values for initial values
np.random.seed(1)
#dividing by 5 to start with smaller weights
w = np.random.random(x_train.shape[1])/5
b = np.random.random(1)/5
print(x_train.shape)
print(b.shape)

parameters = logistic_regression(w, b, x_train, y_train, x_test, y_test, 0.01, 10000)



#for kaggle submission
df = pd.read_csv("titanic/test.csv")
passenger_id = df.PassengerId.values
kaggle_x_test = data_prep(df, True)
predictions = predict(parameters['w'], parameters['b'], kaggle_x_test)
csv = {'PassengerId':passenger_id, "Survived":predictions}
csv = pd.DataFrame(csv)
csv.to_csv('submission.csv' , index=False)