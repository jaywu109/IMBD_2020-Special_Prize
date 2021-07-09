from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PowerTransformer
import math


# 處理 data 套件
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR


def skew_pro(data):
    columns = data.drop(['Predict'], axis=1).columns
    for col in columns:
        if abs(data[col].skew()) >= 0.7:
            pt = PowerTransformer()
            d = pt.fit_transform(data[col].values.reshape(-1, 1)).flatten()
            data[col] = d
    X = data[columns]
    scaler = RobustScaler()
    data[columns] = scaler.fit_transform(X)

    return data


def lr_rmse_ave(x, y):

    train_rmse = []
    test_rmse = []
    test_r2 = []

    for i in np.arange(10):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=i)
        lr = LinearRegression().fit(x_train, y_train)

        y_train_pred = lr.predict(x_train)
        y_test_pred = lr.predict(x_test)
        train_rmse.append(mean_squared_error(
            y_train, y_train_pred, squared=False))
        test_rmse.append(mean_squared_error(
            y_test, y_test_pred, squared=False))
        test_r2.append(r2_score(y_test, y_test_pred))

    train_rmse = np.array(train_rmse).mean()
    test_rmse = np.array(test_rmse).mean()
    test_r2 = np.array(test_r2).mean()

#     print('train_rmse:', train_rmse)
#     print('test_rmse:', test_rmse)
#     print('test_r2:', test_r2)
    return train_rmse, test_rmse, test_r2


def lr_rmse_ave_fea(data, fea_num):

    num = data.select_dtypes(exclude='object')
    numcorr = num.corr()
    cols = abs(numcorr['Predict']).sort_values(ascending=False).head(
        fea_num+1).to_frame().index.to_numpy()[1:]
    y = data['Predict']
    X = data[cols]
    train_rmse = []
    test_rmse = []
    test_r2 = []

    for i in np.arange(10):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, random_state=i)
        lr = LinearRegression().fit(x_train, y_train)

        y_train_pred = lr.predict(x_train)
        y_test_pred = lr.predict(x_test)
        train_rmse.append(mean_squared_error(
            y_train, y_train_pred, squared=False))
        test_rmse.append(mean_squared_error(
            y_test, y_test_pred, squared=False))
        test_r2.append(r2_score(y_test, y_test_pred))

    train_rmse = np.array(train_rmse).mean()
    test_rmse = np.array(test_rmse).mean()
    test_r2 = np.array(test_r2).mean()

    print('train_rmse:', train_rmse)
    print('test_rmse:', test_rmse)
    print('test_r2:', test_r2)
#     return train_rmse, test_rmse, test_r2


def laso_rmse_ave(x, y, alp):
    from sklearn.linear_model import Lasso

    rmse = []
    r2 = []

    for i in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i)
        lasso_mod = Lasso(alpha=alp)
        lasso_mod.fit(X_train, y_train)
        y_lasso_train = lasso_mod.predict(X_train)
        y_lasso_test = lasso_mod.predict(X_test)
        rmse.append(math.sqrt(mean_squared_error(y_test, y_lasso_test)))
        r2.append(r2_score(y_test, y_lasso_test))

    test_rmse = np.array(rmse).mean()
    print('test_rmse_ave:', test_rmse)
    print(rmse)
    print('\n')
    test_r2 = np.array(r2).mean()
    print('test_r2_ave:', test_r2)
    print(r2)


def ElasticNet_rmse_ave(x, y, alp, l1r):
    from sklearn.linear_model import ElasticNet

    rmse = []
    r2 = []

    for i in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i)
        ElasticNet_mod = ElasticNet(alpha=alp, l1_ratio=l1r)
        ElasticNet_mod.fit(X_train, y_train)
        y_ElasticNet_train = ElasticNet_mod.predict(X_train)
        y_ElasticNet_test = ElasticNet_mod.predict(X_test)
        rmse.append(math.sqrt(mean_squared_error(y_test, y_ElasticNet_test)))
        r2.append(r2_score(y_test, y_ElasticNet_test))

    test_rmse = np.array(rmse).mean()
    print('test_rmse_ave:', test_rmse)
    print(rmse)
    print('\n')
    test_r2 = np.array(r2).mean()
    print('test_r2_ave:', test_r2)
    print(r2)


def xgb_ave(x, y, model):

    rmse = []
    r2 = []

    for i in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse.append(math.sqrt(mean_squared_error(y_test, y_pred)))
        r2.append(r2_score(y_test, y_pred))

    test_rmse = np.array(rmse).mean()
    print('test_rmse_ave:', test_rmse)
    print(rmse)
    print('\n')
    test_r2 = np.array(r2).mean()
    print('test_r2_ave:', test_r2)
    print(r2)

# best params {'C': 1.3, 'epsilon': 0.4, 'gamma': 1e-07, 'kernel': 'linear'}


def svr_rmse_ave(data, fea_num, model):

    num = data.select_dtypes(exclude='object')
    numcorr = num.corr()
    title = abs(numcorr['Predict']).sort_values(ascending=False).head(
        fea_num+1).to_frame().index.to_numpy()[1:]
    y = data['Predict']
    X = data[title]
    train_rmse = []
    test_rmse = []
    test_r2 = []

    for i in np.arange(10):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, random_state=i)
        svr = model.fit(x_train, y_train)
        y_train_pred = svr.predict(x_train)
        y_test_pred = svr.predict(x_test)
        train_rmse.append(mean_squared_error(
            y_train, y_train_pred, squared=False))
        test_rmse.append(mean_squared_error(
            y_test, y_test_pred, squared=False))
        test_r2.append(r2_score(y_test, y_test_pred))

    train_rmse = np.array(train_rmse).mean()
    test_rmse = np.array(test_rmse).mean()
    test_r2 = np.array(test_r2).mean()

    print('train_rmse:', train_rmse)
    print('test_rmse:', test_rmse)
    print('test_r2:', test_r2)
#     return train_rmse, test_rmse, test_r2


def rand_ave(x, y, model):

    rmse = []
    r2 = []

    for i in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse.append(math.sqrt(mean_squared_error(y_test, y_pred)))
        r2.append(r2_score(y_test, y_pred))

    test_rmse = np.array(rmse).mean()
    print('test_rmse_ave:', test_rmse)
    print(rmse)
    print('\n')
    test_r2 = np.array(r2).mean()
    print('test_r2_ave:', test_r2)
    print(r2)


def voting_ave(x, y, vote_mod):

    rmse = []
    r2 = []
    mode = vote_mod
    for i in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i)

        vote = mode.fit(X_train, y_train)
        vote_pred = vote.predict(X_test)

        rmse.append(math.sqrt(mean_squared_error(y_test, vote_pred)))
        r2.append(r2_score(y_test, vote_pred))

    test_rmse = np.array(rmse).mean()
    print('test_rmse_ave:', test_rmse)
    print(rmse)
    print('\n')
    test_r2 = np.array(r2).mean()
    print('test_r2_ave:', test_r2)
    print(r2)


def stack_ave(x, y, stack_mod):

    rmse = []
    r2 = []
    mode = stack_mod
    for i in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i)

        stack = mode.fit(X_train, y_train)
        stack_pred = stack.predict(X_test)

        rmse.append(math.sqrt(mean_squared_error(y_test, stack_pred)))
        r2.append(r2_score(y_test, stack_pred))

    test_rmse = np.array(rmse).mean()
    print('test_rmse_ave:', test_rmse)
    print(rmse)
    print('\n')
    test_r2 = np.array(r2).mean()
    print('test_r2_ave:', test_r2)
    print(r2)


def ave(x, y, stack_w, stack_mod, vote_w, vote_mod, model_w, model,):

    rmse = []
    r2 = []
    for i in np.arange(10):
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=i)

        stack = stack_mod.fit(X_train, y_train)
        stack_pred = stack.predict(X_test)
        vote = vote_mod.fit(X_train, y_train)
        vote_pred = vote.predict(X_test)
        mod = model.fit(X_train, y_train)
        mod_pred = mod.predict(X_test)

        ###
        final_test = (vote_w*vote_pred+stack_w*stack_pred + model_w*mod_pred)
        ###

        rmse.append(math.sqrt(mean_squared_error(y_test, final_test)))
        r2.append(r2_score(y_test, final_test))

    test_rmse = np.array(rmse).mean()
    print('test_rmse_ave:', test_rmse)
    print(rmse)
    print('\n')
    test_r2 = np.array(r2).mean()
    print('test_r2_ave:', test_r2)
    print(r2)
