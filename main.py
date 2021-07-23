import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from lightgbm import LGBMRegressor

import semi_supervised_learning

import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(df_train_labeled, df_train_unlabeled, df_test, y_train_labeled):
    epsilon = 1e-3

    number_of_labeled_train_samples = df_train_labeled.shape[0]

    df_train = pd.concat([df_train_labeled, df_train_unlabeled])
    df_train = pd.get_dummies(df_train, columns=["League", "Division", "NewLeague"], drop_first=True)
    df_t = pd.get_dummies(df_test, columns=["League", "Division", "NewLeague"], drop_first=True)

    df_train.reset_index(inplace=True)
    df_t.reset_index(inplace=True)

    features_to_derive = ['AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks']

    for feature in features_to_derive:
        df_train[feature + 'Avg'] = df_train['C' + feature] / df_train['Years']
        df_t[feature + 'Avg'] = df_t['C' + feature] / df_t['Years']

        df_train[feature + 'Last'] = df_train[feature] / df_train['C' + feature]
        df_t[feature + 'Last'] = df_t[feature] / df_t['C' + feature]

        df_train[feature + 'Avg'] = np.log(df_train[feature + 'Avg'] + epsilon)
        df_t[feature + 'Avg'] = np.log(df_t[feature + 'Avg'] + epsilon)

        df_train[feature + 'Last'] = np.log(df_train[feature + 'Last'] + epsilon)
        df_t[feature + 'Last'] = np.log(df_t[feature + 'Last'] + epsilon)

    df_train.fillna(value=0, inplace=True)
    df_t.fillna(value=0, inplace=True)

    log_normalized_features = ['Walks', "CAtBat", "CHits", "CHmRun", "CRuns", "CRBI", "CWalks", "PutOuts", "Assists",
                               'Years', 'AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Errors']

    for feature in log_normalized_features:
        df_train[feature] = np.log(df_train[feature] + epsilon)
        df_t[feature] = np.log(df_t[feature] + epsilon)

    # df_train = df_train.drop(columns=['Division_W', 'League_N'])
    # df_t = df_t.drop(columns=['Division_W', 'League_N'])

    t = np.empty(df_train.shape[0])
    t[:] = np.nan
    t[:number_of_labeled_train_samples] = y_train_labeled
    df_train['y'] = t

    imputer = KNNImputer(n_neighbors=3)
    df_train_imputed = imputer.fit_transform(df_train.to_numpy())

    number_of_features = df_train_imputed.shape[1]

    return df_train_imputed[:number_of_labeled_train_samples, :number_of_features - 1], \
           df_train_imputed[number_of_labeled_train_samples:, :number_of_features - 1], \
           df_train_imputed[number_of_labeled_train_samples:, -1], \
           df_t.to_numpy()


def main(name):
    print(name)

    df = pd.read_csv('./data/Hitters.csv')

    # Separate the dataset into two - Labeled and Unlabeled
    df_labeled = df.dropna()
    df_unlabeled = df[df['Salary'].isna()].drop(['Salary'], axis=1)

    # Split the Labeled set into train/test sets
    df_train_labeled, df_test, y_train_labeled, y_test = train_test_split(df_labeled.drop(['Salary'], axis=1),
                                                                          df_labeled['Salary'],
                                                                          test_size=0.2, random_state=42)

    X_train_labeled, X_train_unlabeled, y_train_unlabeled, X_test = preprocess(df_train_labeled=df_train_labeled,
                                                                               df_train_unlabeled=df_unlabeled,
                                                                               df_test=df_test,
                                                                               y_train_labeled=y_train_labeled)

    X_train = np.concatenate((X_train_labeled, X_train_unlabeled), axis=0)
    y_train = np.concatenate((y_train_labeled, y_train_unlabeled))

    y_train_labeled = y_train_labeled.to_numpy()
    y_test = y_test.to_numpy()

    lgbm = LGBMRegressor(boosting_type='dart', num_leaves=32, learning_rate=0.1, n_estimators=90, colsample_bytree=0.5,
                         max_depth=7)

    lgbm.fit(X_train_labeled, y_train_labeled)
    y_train_pred_single = lgbm.predict(X_train_labeled)
    y_pred = lgbm.predict(X_test)

    rmse_train_single = np.sqrt(mean_squared_error(y_train_labeled, y_train_pred_single))
    rmse_test_single = np.sqrt(mean_squared_error(y_test, y_pred))






    lgbm.fit(X_train, y_train)
    y_train_pred = lgbm.predict(X_train_labeled)
    y_pred = lgbm.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train_labeled, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    print('Control Point - 1')

    ssl = semi_supervised_learning.SSL(n_learners=10)
    ssl.fit(X_train_labeled=X_train_labeled, y_train_labeled=y_train_labeled, X_train_unlabeled=X_train_unlabeled)
    y_train_pred_ssl = ssl.predict(X_train_labeled)
    y_pred_ssl = ssl.predict(X_test)

    rmse_train_ssl = np.sqrt(mean_squared_error(y_train_labeled, y_train_pred_ssl))
    rmse_test_ssl = np.sqrt(mean_squared_error(y_test, y_pred_ssl))

    print(f'LGBM SINGLE RMSE Train: {rmse_train_single}')
    print(f'LGBM SINGLE RMSE Test: {rmse_test_single}')

    print(f'LGBM RMSE Train: {rmse_train}')
    print(f'LGBM RMSE Test: {rmse_test}')

    print(f'SSL RMSE Train: {rmse_train_ssl}')
    print(f'SSL RMSE Test: {rmse_test_ssl}')

    ssl.print_params()

    dummy = -32


if __name__ == '__main__':
    main('Kaggle Hitters')
