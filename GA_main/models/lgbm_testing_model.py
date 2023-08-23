import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import lightgbm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import encoder_transformation
import shap

original = pd.read_csv('data/original_preprocessed.csv')
additional = pd.read_csv('data/additional_preprocessed.csv')
test = pd.read_csv('data/final_test.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)



transfo = encoder_transformation.data_encode(all)
X = transfo
Y = all[['viability (%)']].copy()

tested = encoder_transformation.data_encode(test)
X_val = tested
Y_val = test[['viability (%)']].copy()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
# pickle.dump(model, open('lgbm_model_final.pkl', 'wb'))
train = model.predict(X_train)
validation = model.predict(X_test)
testing =model.predict(X_val)

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
sns.set(font_scale=2)
f, ax = plt.subplots(figsize=(13, 10))
plt.scatter(Y_train, train, color='#DD7059', s=40, label='train data')  # , alpha=0.5)
plt.scatter(Y_test, validation, color='#569FC9', s=40, label='validation')  # , alpha= 0.5)
# plt.scatter(Y_val, testing, color='#274E13',s=70, label = 'testing')
plt.plot(Y_test, Y_test, color='#444444', linewidth=3)
# plt.plot(Y_test, (Y_test - np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
# plt.plot(Y_test, (Y_test + np.sqrt(metrics.mean_squared_error(Y_test, validation))), color='#444444', linewidth = 1)
plt.title('Optimized LGBM Regressor')
plt.xlabel('actual data')
plt.ylabel('predicted data')
plt.legend()
plt.xlim(0, 135)
plt.ylim(0, 135)

# set axis line width and color
ax.spines['bottom'].set_linewidth(2)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_color('black')

plt.show()
# ax.figure.savefig("opt_LGBM_regressor.png", transparent=True, )

X_importance = X_test
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)
shap.summary_plot(shap_values, X_importance, show=False)
plt.figure(figsize= (15,10))
plt.show()
# plt.savefig('important_features.png')

# feature importance
importance = model.feature_importances_
features = model.feature_name_
fig4 = plt.figure(constrained_layout=True, figsize=(10, 6))
indices = np.argsort(importance)
feature_dict = {key: value for key, value in zip(features, importance)}
feature_dictionary = pd.DataFrame(zip(features, importance * 100), columns=["Feature ID", "Importance"])
feature_dictionary.sort_values(by="Importance", ascending=False, inplace=True)
feature_dictionary = feature_dictionary.head(5)
plt.ylabel("Feature ID", fontsize=16, labelpad=20)
plt.xlabel("Importance", fontsize=16)
plt.title("LGBMRegressor feature importance", fontsize=20)
sns.set_palette("rocket")
sns.barplot(feature_dictionary, x="Importance", y="Feature ID", edgecolor='black', linewidth=1)
plt.show()
# plt.savefig('feature_lgbm.png')
