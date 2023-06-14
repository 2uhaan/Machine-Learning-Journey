import pandas as pd  #for datasets
import numpy as np   #for Math
import matplotlib as plt  #for Viz

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#Catogorical data to numeric data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

#dummy encryption
one = OneHotEncoder()
encode = one.fit_transform(x[:, 3].reshape(-1, 1)).toarray()
x = np.delete(x, 0, axis=1)
x = np.concatenate((encode, x), axis=1)

x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

#Append constant b0 in x
x = np.append(arr= np.ones((50,1)).astype(int), values = x, axis=1)

#Backward Elimination
import statsmodels.api as sm
x_opt = x[:, [0,1,2,3,4,5]]
regrassor_OLS = sm.OLS(y, x_opt.astype(float)).fit()
regrassor_OLS.summary()

#p-value greater than 0.05 ( 5 % )  Remove Manually
x_opt = x[:, [0,1,2,3,4]]
regrassor_OLS = sm.OLS(y, x_opt.astype(float)).fit()
regrassor_OLS.summary()


x_opt = x[:, [0,2,3,4]]
regrassor_OLS = sm.OLS(y, x_opt.astype(float)).fit()
regrassor_OLS.summary()


x_opt = x[:, [0,3,4]]
regrassor_OLS = sm.OLS(y, x_opt.astype(float)).fit()
regrassor_OLS.summary()


x_opt = x[:, [3,4]]
regrassor_OLS = sm.OLS(y, x_opt.astype(float)).fit()
regrassor_OLS.summary()


