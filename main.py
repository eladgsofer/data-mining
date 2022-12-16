
# imports
import pandas as pd
import numpy as np
import statsmodels.api as sm

# functions
def normalize_col(df, col_name, scaled_flag=True):
    if scaled_flag:
        max_val = max(df[col_name])
        df[col_name] = df[col_name] / max_val
        return 0, max_val
    else:
        mean = np.mean(df[col_name])
        std = np.std(df[col_name])
        df[col_name] = (df[col_name] - mean) / std
        return mean,std


def data_processing_train(df):
    df = df[df['Life expectancy '].notna()]    # drop lines with NaN in Life expectancy

    df.Status = [1 if stat=="Developed" else 0 for stat in df.Status] # make Status binary, Developed=1, Developing=0

    df["Year"] = (df["Year"]-2000)/15   # reducing the years value

    scaling_factors = []     # list of all the scaling factor of the the floats columns
    for i in range(3, 22):      # normalize all the float columns
        col_name = df.columns[i]
        scaling_factors.append(normalize_col(df, col_name, scaled_flag=True))

    df.bfill(inplace=True)   # fill empty cells with the next cell value

    country_code = pd.get_dummies(df['Country'])
    df.drop(['Country'], axis=1, inplace=True)
    df = pd.concat([df, country_code], axis=1)
    return df, scaling_factors, country_code.columns


def data_processing_test(df, scale_factor, country_names):
    df.drop(['ID'], axis=1, inplace=True)   # drop the ID column

    df.Status = [1 if stat=="Developed" else 0 for stat in df.Status] # make Status binary, Developed=1, Developing=0

    df["Year"] = (df["Year"]-2000)/15   # reducing the years value

    for i in range(3, 21):      # normalize all the float columns
        df.iloc[:,i] = (df.iloc[:,i]-scale_factor[i-2][0])/scale_factor[i-2][1]

    df.bfill(inplace=True)   # fill empty cells with the next cell value

    country_code = pd.DataFrame(data=np.zeros((len(df),len(country_names))), columns=country_names)
    for i in range(len(df)):
        country_code.loc[i][df.loc[i]['Country']] = 1
    df.drop(['Country'], axis=1, inplace=True)
    df = pd.concat([df, country_code], axis=1)
    return df


# make all the columns be represented when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# read df
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ID = test['ID']
train, scale_factors, country_names = data_processing_train(train)
print(train.head())
print(train.describe())

y_train = np.array(train['Life expectancy '])
train.drop(['Life expectancy '], axis=1, inplace=True)
x_train = train.to_numpy()

#fit linear regression model
model = sm.OLS(y_train, x_train).fit()

#view model summary
print(model.summary())

# predict
x_test = data_processing_test(test, scale_factors, country_names)
y_test = model.predict(x_test)
res = pd.DataFrame()
res['ID'] = ID
res['Life expectancy'] = y_test*scale_factors[0][1]+scale_factors[0][0]
res.to_csv('test_res2.csv', index=False)


