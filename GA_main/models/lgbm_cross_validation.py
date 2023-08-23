import pandas as pd
import numpy as np
import lightgbm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

import encoder_transformation

original = pd.read_csv('data/original_preprocessed.csv')
additional = pd.read_csv('data/additional_preprocessed.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)


transfo = encoder_transformation.data_encode(all)
X = transfo
Y = all[['viability (%)']].copy()


train_R2_metric_results = []
train_mse_metric_results = []
train_mae_metric_results = []
test_R2_metric_results = []
test_mse_metric_results = []
test_mae_metric_results = []
cv = StratifiedShuffleSplit(n_splits=10, test_size = 0.2, random_state = 42)
cv_scores = np.empty(10)
for idx, (train_indices, test_indices) in enumerate(cv.split(X, X[['amw']])):
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
    model = lightgbm.LGBMRegressor(n_estimators=778,
                                   learning_rate=0.07708799000898921,
                                   num_leaves=250,
                                   max_depth=12,
                                   min_data_in_leaf=20,
                                   lambda_l1=6.445750753675395,
                                   lambda_l2=6.1260719233610645,
                                   bagging_fraction=0.9789495809994003,
                                   bagging_freq=1,
                                   feature_fraction=0.8571741750071638, )

    lgb_model = model.fit(X_train, Y_train)
    train = model.predict(X_train)
    validation = model.predict(X_test)

    train_R2_metric_results.append(r2_score(Y_train, train))
    train_mse_metric_results.append(mean_squared_error(Y_train, train))
    train_mae_metric_results.append(mean_absolute_error(Y_train, train))

    test_R2_metric_results.append(r2_score(Y_test, validation))
    test_mse_metric_results.append(mean_squared_error(Y_test, validation))
    test_mae_metric_results.append(mean_absolute_error(Y_test, validation))

print('Train')
print('Train R-square:', np.mean(train_R2_metric_results))
print('Mean Absolute Error:', np.mean(train_mae_metric_results))
print('Mean Squared Error:', (np.mean(train_mse_metric_results)))
print('Root Mean Squared Error:', (np.mean(train_mse_metric_results)**(1/2)))

print('validation')
print('one-out cross-validation (R-square):', r2_score(Y_test, validation))
print('10-fold cross-validation result (R-square):', np.mean(test_R2_metric_results))
print('10-fold cross-validation result (MAE):', np.mean(test_mae_metric_results))
print('10-fold cross-validation result (MSE):', (np.mean(test_mse_metric_results)))
print('10-fold cross-validation result (RMSE):', (np.mean(test_mse_metric_results)**(1/2)))


"""#Visualization"""

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(13, 10))
plt.scatter(Y_train, train, color='#DD7059', s=70, label = 'train data')#, alpha=0.5)
plt.scatter(Y_test, validation , color='#569FC9',s=70, label = 'validation')#, alpha= 0.5)
plt.plot(Y_test, Y_test, color='#444444', linewidth=3)
plt.title('LGBM Regressor')
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.xlim(0, 135)
plt.ylim(0, 135)
plt.show()
# ax.figure.savefig("LGBM_regressor.png",transparent=True)