# kaggle_hitters

https://www.kaggle.com/floser/hitters/



References:

- https://www.kaggle.com/lmorgan95/islr-tree-based-methods-ch-8-solutions
- https://www.kaggle.com/faelk8/hiters-eda-multiple-models/notebook
- https://www.kaggle.com/serhatyzc/salary-prediction-using-the-lightgbm-algorithm
- https://www.kaggle.com/ahmetcankaraolan/salary-predict-with-nonlinear-regression-models/ (Salary comment)
- https://www.kaggle.com/nguncedasci/rmse-183-9-hitters-data-set-light-gbm-modeling

- https://stats.stackexchange.com/questions/447863/log-transforming-target-var-for-training-a-random-forest-regressor
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html



Notes:

* Log-transform the target variable (due to skewness)
* LabelEncoder should be used to encode target values, i.e. y, and not the input X.
* Function to separate variables in data frame as categorical and numerical 
** https://www.kaggle.com/serhatyzc/salary-prediction-using-the-lightgbm-algorithm?scriptVersionId=59290598&cellId=13
* Function setting an upper and lower limit for distribution of each attribute
** https://www.kaggle.com/serhatyzc/salary-prediction-using-the-lightgbm-algorithm?scriptVersionId=59290598&cellId=16

* For a unimodal distribution, negative skew commonly indicates that the tail is on the left side of the distribution, and positive skew indicates that the tail is on the right. In cases where one tail is long but the other tail is fat, skewness does not obey a simple rule. For example, a zero value means that the tails on both sides of the mean balance out overall; this is the case for a symmetric distribution, but can also be true for an asymmetric distribution where one tail is long and thin, and the other is short but fat.

* LocalOutlierFactor
** https://www.kaggle.com/nguncedasci/rmse-183-9-hitters-data-set-light-gbm-modeling?scriptVersionId=38819099&cellId=46
