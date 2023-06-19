
import pandas as pd  
import numpy as np   
from matplotlib import pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
poly_x = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_x, y)

plt.scatter(x, y)
plt.plot(x, lin_reg.predict(x))
plt.title('Linear Regression')
plt.show()

x_grid = np.arange(min(x), max(x), 0.2)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y)
# plt.plot(x_grid, poly_reg.fit_transform(x_grid))
plt.plot(x_grid, lin_reg_2.predict(poly_reg.transform(x_grid)))
plt.title('Polynomial Regression')
plt.show()

lin_reg.predict(6.5)
lin_reg_2.predict(poly_reg.fit_transform(6.5))
