import pandas as pd  #for datasets
import numpy as np   #for Math
import matplotlib as plt  #for Viz

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, random_state=0)

#Model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

#sample Predictions
y_pred = reg.predict(x_test)

#Plot for Train
plt.pyplot.scatter(x_train, y_train, color = 'red')
plt.pyplot.plot(x_train, reg.predict(x_train), color = 'blue')
plt.pyplot.title('Salary vs Exp')
plt.pyplot.xlabel('Exp')
plt.pyplot.ylabel('Salary')
plt.pyplot.show()

#Plot for Test
plt.pyplot.scatter(x_test, y_test, color = 'blue')
plt.pyplot.plot(x_train, reg.predict(x_train), color = 'red')
plt.pyplot.title('Salary vs Exp - Test')
plt.pyplot.xlabel('Exp')
plt.pyplot.ylabel('Salary')
plt.pyplot.show()