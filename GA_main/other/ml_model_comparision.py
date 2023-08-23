import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

import encoder_transformation

original = pd.read_csv('data/original_preprocessed.csv')
additional = pd.read_csv('data/additional_preprocessed.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)
raw = pd.read_csv('data/original_additional_raw_all_descriptors.csv')

transfo = encoder_transformation.data_encode(all)
trans_raw = encoder_transformation.data_encode(raw)
X = transfo
Y = all[['viability (%)']].copy()
Xraw = trans_raw
Yraw = raw[['viability (%)']].copy()


#change the value of X, Y for training and testing
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(Xraw, Yraw, test_size=0.2, random_state=42)

# Defines and builds the lazyclassifier
clf = LazyRegressor(verbose=2 ,ignore_warnings=False, custom_metric=None)
#sometime it can get stucked check for QuantileRegressor, it might be creating problem
train,test = clf.fit(X_train, X_test, Y_train, Y_test)

train_mod = train.iloc[: , :]
train_mod.to_csv('results/lazypredict/Model_comparision_train.csv')
print(train_mod)
test_mod = test.iloc[: , :]
test_mod.to_csv('results/lazypredict/Model_comparision_test.csv')
# print(test_mod)


"""#Data Visualization

"""

# Bar plot of R-squared values
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train_mod.index, x="R-Squared", data=train_mod)
ax.set(xlim=(0, 1))

# Bar plot of RMSE values
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train_mod.index, x="RMSE", data=train_mod)
ax.set(xlim=(0, 1))

# Bar plot of calculation time
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=train.index, x="Time Taken", data=train)
ax.set(xlim=(0, 5))
plt.show()