import lightgbm as lgb
import pandas as pd
from lightgbm import Dataset
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
import encoder_transformation

original = pd.read_csv('data/original_preprocessed.csv')
additional = pd.read_csv('data/additional_preprocessed.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)


#save transformed data
def trans_saver(data):
    trans_X = encoder_transformation.data_encode(data) #this is X value
    trans_Y = data['viability (%)'].copy()
    trans_org = pd.concat([trans_X, trans_Y], axis=1)
    trans_org.to_csv('transformed_data.csv')
    return  trans_org

# Define the objective function
def objective(trial):
    transfo = encoder_transformation.data_encode(all)
    X = transfo
    Y = all[['viability (%)']].copy()

    # Create the splits
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    # Convert to LightGBM Dataset
    for train_index, test_index in sss.split(X, X[['amw']]):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        train_data = Dataset(data=X_train, label=y_train)
        test_data = Dataset(data=X_test, label=y_test)

    # Define the parameters
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
        'early_stopping_round': 10
    }

    # Train the model
    model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[train_data, test_data], verbose_eval=False)
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


# Run the Optuna study
study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Print the best hyperparameters found by Optuna
print("Best hyperparameters: ", study.best_params)
