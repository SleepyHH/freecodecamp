# Linear Regression y = B0 + B1*X
# error function e*i = y*i - ybar*i

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

%matplotlib inline

data = pd.read_csv('data/Advertising.csv',index_col=0)
data.head()

# Simple linear regression
# 
plt.figure(figsize=(16,8)) # figure size
plt.scatter(data['TV'], data ['sales'],c='black') #scatterplot
plt.xlabel =('Money spent on TV ads ($)')
plt.ylabel('Sales (k$)')
plt.show()

X = data ['TV'].values.reshape(-1,1) #specify feature TV
y = data['sales']balues.reshape(-1,1) #specify target

reg = LinearRegression()                # initialise
reg.fit = (X,y)                         # fit on x and y

print(f"The linear model is \n Y = {reg.incercept_[0]+ reg.cpef_[0][0]}*TV")
intercept = reg.intercept_[0]           # retrieve incercept and coeffcient
coef = reg.coef_[0][0]

# assess relevancy using p-value to quantify statistical significance to determine rejection of null hypothesis or not

# statistical package to print out summary of model
exog =  sm.add_constant(X)
est = sm.0LS(y , exog).fit()
print(est.summary())