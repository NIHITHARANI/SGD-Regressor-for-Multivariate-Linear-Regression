# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Step-1.Start
Step-2.Data Preparation
3.Hypothesis Definition
4.Cost Function
5.Parameter Update Rule
6.Iterative Training
7.Model Evaluation
8.End 

## Program:
```py
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: B.NIHITHA RANI 
RegisterNumber:  212223040131
*/
import numpy as np 
from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X= data.data[:, :3]
Y = np.column_stack((data.target, data.data[:, 6]))
x_train, x_test, Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state =42)
scaler_X=StandardScaler()
scaler_Y= StandardScaler()
x_train = scaler_X.fit_transform(x_train)
x_test = scaler_X.fit_transform(x_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.fit_transform(Y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, Y_train)
y_pred =multi_output_sgd.predict(x_test)
y_pred = scaler_Y.inverse_transform(y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
print(y_pred)
mse = mean_squared_error(Y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
```

## Output:
```
y_pred
```
![6-1](https://github.com/user-attachments/assets/2ec20159-89a2-47cc-ab42-43c1add6ae36)

```
Mean squared error 
```
![6-2](https://github.com/user-attachments/assets/c35ea309-2b26-474b-9261-ec67043526e6)

```
y_prediction
```
![6-3](https://github.com/user-attachments/assets/1bf43ec9-f52f-409a-b8e8-bfbc0aaa4760)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
