
# imports
import pandas as pd
import numpy as np
import statsmodels.api as sm

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.drop(['Year'], axis=1, inplace=True)
train.drop([' thinness 5-9 years'], axis=1, inplace=True)
train.drop(['Population'], axis=1, inplace=True)
train.drop(['percentage expenditure'], axis=1, inplace=True)
train.drop(['Total expenditure'], axis=1, inplace=True)
train.drop(['Country'], axis=1, inplace=True)
train.drop(['Status'], axis=1, inplace=True)
train = train.dropna(axis=0)

y = np.array(train['Life expectancy '])
train.drop(['Life expectancy '], axis=1, inplace=True)
x = train[train.columns]
x = np.array(sm.add_constant(x))

#fit linear regression model
model = sm.OLS(y, x).fit()


#view model summary
print(model.summary())