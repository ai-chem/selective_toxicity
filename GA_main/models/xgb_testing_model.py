import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap
import encoder_transformation

original = pd.read_csv('data/original_preprocessed.csv')
additional = pd.read_csv('data/additional_preprocessed.csv')
test = pd.read_csv('data/final_test.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)

transfo = encoder_transformation.data_encode(all)
X = transfo
Y = all[['viability (%)']].copy()
X_val = encoder_transformation.data_encode(test)
Y_val = test[['viability (%)']].copy()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=884,
                       learning_rate=0.035958915605443356,
                       max_depth=8,
                       min_child_weight= 3,
                       subsample= 0.8845398751174367,
                       colsample_bytree = 0.9852832952114663,
                       reg_lambda = 0.0013221099443591067,
                       reg_alpha = 3.16693837876956,
                     )

xgb_model = model.fit(X_train, Y_train)
train = model.predict(X_train)
validation = model.predict(X_test)

testing = model.predict(X_val)

print('Train')
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_train, train))
print('Mean Squared Error:', metrics.mean_squared_error(Y_train, train))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_train, train)))
print("Regressor R2-score: ", r2_score(Y_train, train))

print('validation')
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, validation))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, validation))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, validation)))
print("Regressor R2-score: ", r2_score(Y_test, validation))

print('testing')
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_val, testing))
print('Mean Squared Error:', metrics.mean_squared_error(Y_val, testing))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_val, testing)))
print("Regressor R2-score: ", r2_score(Y_val, testing))


"""#Visualization"""
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(13, 10))
plt.scatter(Y_train, train, color='#DD7059', s=70, label = 'train data')#, alpha=0.5)
plt.scatter(Y_test, validation , color='#569FC9',s=70, label = 'validation')#, alpha= 0.5)
plt.scatter(Y_val, testing, color='#274E13',s=70, label = 'testing')
plt.plot(Y_test, Y_test, color='#444444', linewidth=3)
plt.plot(Y_test, (Y_test - 2*np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
plt.plot(Y_test, (Y_test + 2*np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
# sns.regplot(x=Y_test, y=validation, ci=0.95, color='#274E13')
plt.title('XGB Regressor')
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.xlim(0, 135)
plt.ylim(0, 135)
plt.show()
# ax.figure.savefig("LGBM_regressor.png",transparent=True)

X_importance = X_test
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance, show=False)
plt.figure(figsize= (15,10))
plt.show()
# plt.savefig('important_features.png')

