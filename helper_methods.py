import pandas as pd
import statsmodels.api as sm
import math


def get_data():
    data = pd.read_csv('./resources/project-data.csv')
    development_data_raw = data[~data['SalePrice'].isnull()]
    evaluation_data_raw = data[data['SalePrice'].isnull()]

    development_data = development_data_raw.dropna()
    evaluation_data = evaluation_data_raw.drop(columns=['SalePrice'])
    evaluation_data.dropna()
    categorical_feature_names = ['MSZoning', 'Street', 'Utilities', 'BldgType', 'BsmtQual', 'ExterQual', 'ExterCond',
                                 'Heating', 'GarageCond']
    development_data = label_categorical_features(development_data, categorical_feature_names)
    evaluation_data = label_categorical_features(evaluation_data, categorical_feature_names)

    evaluation_data.drop(columns=['Id'], inplace=True)
    development_data.drop(columns=['Id'], inplace=True)
    return [development_data, evaluation_data]


def split_development_data(development_data):
    training_set = development_data.sample(frac=0.8)
    testing_set = development_data.drop(index=training_set.index)
    trainX = training_set.drop(columns=['SalePrice'])
    trainY = training_set['SalePrice']
    testX = testing_set.drop(columns=['SalePrice'])
    testY = testing_set['SalePrice']
    return [trainX, trainY, testX, testY]


def transformation_of_dev_data(data):
    return data[['LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'GarageCars', 'ExterQual_TA', 'SalePrice']]


def transformation_of_eval_data(data):
    return data[['LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'GarageCars', 'ExterQual_TA']]


def label_categorical_features(data_set, column_names):
    return pd.get_dummies(data=data_set, prefix=column_names)


# RMSE
def root_mean_square_error(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise Exception('Vectors do not have the same type')
    diff = y_pred - y_true
    return math.sqrt(diff.T.dot(diff)/len(diff))


# MAE
def mean_absolute_error(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise Exception('Vectors do not have the same type')
    diff = (y_pred - y_true).apply(lambda x: math.fabs(x))
    return diff.mean()