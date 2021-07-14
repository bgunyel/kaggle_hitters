import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
import seaborn as sns


def preprocess(df_train_labeled, df_train_unlabeled, df_test):
    epsilon = 1

    number_of_labeled_train_samples = df_train_labeled.shape[0]

    df_train = pd.concat([df_train_labeled, df_train_unlabeled])
    df_train = pd.get_dummies(df_train, columns=["League", "Division", "NewLeague"], drop_first=True)
    df_t = pd.get_dummies(df_test, columns=["League", "Division", "NewLeague"], drop_first=True)

    log_normalized_features = ['Walks', "CAtBat", "CHits", "CHmRun", "CRuns", "CRBI", "CWalks", "PutOuts", "Assists",
                               'Years', 'AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Errors']

    for feature in log_normalized_features:
        df_train[feature] = np.log(df_train[feature] + epsilon)
        df_t[feature] = np.log(df_t[feature] + epsilon)

    dummy = -32

    return df_train.iloc[:number_of_labeled_train_samples].to_numpy(), \
           df_train.iloc[number_of_labeled_train_samples:].to_numpy(), \
           df_t.to_numpy()


def main(name):
    print(name)

    df = pd.read_csv('./data/Hitters.csv')

    # Separate the dataset into two - Labeled and Unlabeled
    df_labeled = df.dropna()
    df_unlabeled = df[df['Salary'].isna()].drop(['Salary'], axis=1)

    # Split the Labeled set into train/test sets
    df_train_labeled, df_test, y_train, y_test = train_test_split(df_labeled.drop(['Salary'], axis=1),
                                                                  df_labeled['Salary'],
                                                                  test_size=0.2, random_state=42)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    X_train_labeled, X_train_unlabeled, X_test = preprocess(df_train_labeled=df_train_labeled,
                                                            df_train_unlabeled=df_unlabeled, df_test=df_test)

    lgbm = LGBMRegressor(boosting_type='dart', num_leaves=32, learning_rate=0.1, n_estimators=100, colsample_bytree=0.5,
                         max_depth=7)

    lgbm.fit(X_train_labeled, y_train)
    y_pred = lgbm.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(rmse)

    dummy = -32


if __name__ == '__main__':
    main('Kaggle Hitters')
