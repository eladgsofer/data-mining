import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn import svm
import statsmodels.api as sm

# paths = []
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         paths.append(os.path.join(dirname, filename))
#         print(os.path.join(dirname, filename))

# sample = pd.read_csv(paths[0])
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.columns


#country embeding
# Find unique values of a column
train_for_embd = train
train_for_embd = train_for_embd.sort_values(by=['Life expectancy '],ascending=False)


all_counties = train_for_embd['Country'].unique().tolist()
country_2_num = {}
for i,country in enumerate(all_counties):
    country_2_num.update({country: i})
#country_2_num

dev_stat = train_for_embd['Status'].unique().tolist()
Status_2_num = {}
for i,stat in enumerate(dev_stat):
     Status_2_num.update({stat: i})
#Status_2_num

train["Country"] = train["Country"].map(country_2_num)
test["Country"] = test["Country"].map(country_2_num)

train["Status"] = train["Status"].map(Status_2_num)
test["Status"] = test["Status"].map(Status_2_num)

for cul in train.columns:
    train[cul] = train[cul].astype(float)
for cul in test.columns:
    test[cul] = test[cul].astype(float)



# #fill with avg
# train[ "Status"] = train[ "Status"].fillna(0)
# test[ "Status"] = test[ "Status"].fillna(0)

# train = train[train["Life expectancy "].notna()]


# for column in train:
#     if column != "Status" and column != "Life expectancy ":
#         avg = train[column].median()
#         max_ = train[column].max()
#         min_ = train[column].min()
#         norme = max_ - min_
#         train[column] = train[column].fillna(avg)
#         train[column] = train[column] - min_
#         train[column] = train[column].div(norme)

# for column in test:
#     if column != "ID" and column != "Status" :
#         avg = test[column].median()
#         max_ = test[column].max()
#         min_ = test[column].min()
#         n…

#fill with avg
train[ "Status"] = train[ "Status"].fillna(0)
test[ "Status"] = test[ "Status"].fillna(0)

train = train[train["Life expectancy "].notna()]


for column in train:
    if column != "Status" and column != "Life expectancy ":
        avg = train[column].median()
        train[column] = train[column].fillna(avg)
        train[column] = (train[column] - train[column].mean())/train[column].std()

for column in test:
    if column != "ID" and column != "Status" :
        avg = test[column].median()
        test[column] = test[column].fillna(avg)
        test[column] = (test[column] - test[column].mean())/test[column].std()


mean = train["Life expectancy "].mean()
std = train["Life expectancy "].std()
train["Life expectancy "] = (train["Life expectancy "] - mean).div(std)
train["Life expectancy "] = train["Life expectancy "] + np.random.normal(0, 0.01)


# train.to_csv('check_train.csv',index=False)
# test.to_csv('check_test.csv',index=False)

X = train.drop(columns=['Life expectancy '])
y = train['Life expectancy ']

X.columns.tolist()
#X.dtypes

krr = GridSearchCV(
    KernelRidge(),
    param_grid={"alpha": [2e0, 2.1, 5, 10,1e0, 0.1, 1e-2, 1e-3], 'kernel':('linear','polynomial','laplacian')},
)

krr.fit(X, y)
krr.get_params()


print(len(test['ID']))
IDs = test['ID'].reset_index()
test = test.drop(columns = ['ID'])


test.columns.tolist()
#test.dtypes

predict = krr.predict(test)
predict = {'Life expectancy ': predict}
predict = pd.DataFrame(predict)
print(predict.shape,IDs.shape)
# predict

print(predict.shape, IDs.shape)

IDs = IDs.drop(columns = ['index']).astype(int)
IDs = IDs['ID'].values.tolist()

predict = predict['Life expectancy '].values.tolist()

result  = pd.DataFrame({
    'ID': IDs,
    'Life expectancy': predict,
})

#de_normelize max min
#result['Life expectancy'] = (result['Life expectancy'] *(max_ - min_))+min_

#de_normelize mean,std
result['Life expectancy'] = (result['Life expectancy'] *(std))+mean

result.to_csv('submission.csv',index=False)