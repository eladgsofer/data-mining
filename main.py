
# imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pycountry_convert as pc




def normalize_col(df, col_name, scaled_flag=True):
    if scaled_flag:
        df[col_name] = df[col_name] / max(df[col_name])
        return max(df[col_name]), max(df[col_name])
    else:
        mean = np.mean(df[col_name])
        std = np.std(df[col_name])
        df[col_name] = (df[col_name] - mean) / std
        return mean,std




# make all the columns be represented when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# read df
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.head())
print(train.describe())

# data processing
train = train[train['Life expectancy '].notna()]    # drop lines with NaN in Life expectancy
print(train.describe())
train.Status = [1 if stat=="Developed" else 0 for stat in train.Status] # make Status binary, Developed=1, Developing=0

scaling_factor = []     # list of all the scaling factor of the the floats columns
for i in range(3, 22):      # normalize all the float columns
    col_name = train.columns[i]
    scaling_factor.append(normalize_col(train, col_name))



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