import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


class SSL:
    def __init__(self, n_learners=3):
        self.n_learners = n_learners

        self.lgbm_params = {'boosting_type': 'dart', 'num_leaves': 32, 'learning_rate': 0.1, 'n_estimators': 90,
                            'colsample_bytree': 0.5, 'max_depth': 7}

        self.n_iterations = 5
        self.K = 5

        self.rmse = np.zeros(n_learners)
        self.lgbm = []

        for i in range(n_learners):
            boosting_type = self.lgbm_params['boosting_type']
            num_leaves = np.random.randint(self.lgbm_params['num_leaves'] - n_learners,
                                           self.lgbm_params['num_leaves'] + n_learners + 1)
            n_estimators = np.random.randint(self.lgbm_params['n_estimators'] - 10,
                                             self.lgbm_params['n_estimators'] + 11)
            max_depth = np.random.randint(self.lgbm_params['max_depth'] - 2, self.lgbm_params['max_depth'] + 3)

            learning_rate = np.random.normal(loc=self.lgbm_params['learning_rate'], scale=0.05)
            colsample_bytree = np.random.normal(loc=self.lgbm_params['colsample_bytree'], scale=0.1)

            self.lgbm.append(
                LGBMRegressor(boosting_type=boosting_type, num_leaves=num_leaves, learning_rate=learning_rate,
                              n_estimators=n_estimators, colsample_bytree=colsample_bytree,
                              max_depth=max_depth))

    def fit(self, X_train_labeled, y_train_labeled, X_train_unlabeled):

        X_train = []
        X_val = []
        y_train = []
        y_val = []

        is_available = []

        number_of_unlabeled_samples = X_train_unlabeled.shape[0]

        for i in range(self.n_learners):
            X_t, X_v, y_t, y_v = train_test_split(X_train_labeled, y_train_labeled, test_size=0.2)
            X_train.append(X_t)
            X_val.append(X_v)
            y_train.append(y_t)
            y_val.append(y_v)

            self.lgbm[i].fit(X_t, y_t)
            y_pred = self.lgbm[i].predict(X_v)
            self.rmse[i] = np.sqrt(mean_squared_error(y_v, y_pred))
            is_available.append(np.ones(number_of_unlabeled_samples, dtype=bool))

        for iteration in range(self.n_iterations):
            for i in range(self.n_learners):

                continue_training = True
                k = self.K

                while continue_training and (k > 0):

                    y_pred = np.zeros((number_of_unlabeled_samples, self.n_learners - 1))
                    counter = 0
                    for j in range(self.n_learners):
                        if j != i:
                            y_pred[:, counter] = self.lgbm[j].predict(X_train_unlabeled)
                            counter += 1
                    y_pred_unlabeled = np.mean(y_pred, axis=1)
                    p = y_pred / np.reshape(np.sum(y_pred, axis=1), (-1, 1))
                    h = np.sum(p * np.log(p), axis=1)  # h is not negated and it is sorted in ascending order
                    idx = np.argsort(h)
                    temp_available = is_available[i][idx]
                    temp_cumsum = np.cumsum(temp_available)
                    X_temp = X_train_unlabeled[idx, :]
                    y_temp = y_pred_unlabeled[idx]

                    location = np.where(temp_cumsum == k)[0][0]
                    X_t = np.concatenate((X_train[i], X_temp[:location+1, :]), axis=0)
                    y_t = np.concatenate((y_train[i], y_temp[:location+1]))

                    params = self.lgbm[i].get_params(deep=True)
                    lgbm_t = LGBMRegressor().set_params(**params)
                    lgbm_t.fit(X_t, y_t)
                    y_val_predict = lgbm_t.predict(X_val[i])
                    rmse_t = np.sqrt(mean_squared_error(y_val[i], y_val_predict))

                    if rmse_t < self.rmse[i]:
                        X_train[i] = X_t
                        y_train[i] = y_t

                        params = self.lgbm_t.get_params(deep=True)
                        self.lgbm[i].set_params(**params)
                        continue_training = False
                    else:
                        k -= 1


                    dummy = -32

    def predict(self):
        pass
