# importing Libraries
import pandas as pd  #for datasets
import numpy as np   #for Math
import matplotlib as plt  #for Viz

# read dataset
dataset = pd.read_csv('Data.csv')

#Spliting Dependent (y) and Independent variables (x)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#missing Value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer( missing_values= np.nan , strategy="mean")
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Catogorical data to numeric data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

#dummy encryption
one = OneHotEncoder()
encode = one.fit_transform(x[:, 0].reshape(-1, 1)).toarray()
x = np.delete(x, 0, axis=1)
x = np.concatenate((encode, x), axis=1)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Spliting Train / Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

#Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
