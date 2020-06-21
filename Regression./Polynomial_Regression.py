import numpy as np #math fns
import matplotlib.pyplot as plt #plotting
import pandas as pd #import datasets

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#linReg.predict([[6.5]]) 
#linReg2.predict(polyReg.fit_transform([[6.5]]))

'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)'''

# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,y)
linReg.predict([[6.5]])

#fitting to polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
linReg2 = LinearRegression()
linReg2.fit(X_poly,y)

#Visualizing the linaer regression results
plt.scatter(X,y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'Blue')
plt.title("Truth or bluff(Linear regression)")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()

#Visualizing the polynomial regression results
plt.scatter(X,y, color = 'red')
plt.plot(X, linReg2.predict(poly_reg.fit_transform(X)), color = 'Blue')
plt.title("Truth or bluff(Polynomial regression)")
plt.xlabel("postion level")
plt.ylabel("salary")
plt.show()

#predict a new result with linear regression
linReg.predict([[6.5]])

#predict a new result with polynomial regression
linReg2.predict(poly_reg.fit_transform([[6.5]]))