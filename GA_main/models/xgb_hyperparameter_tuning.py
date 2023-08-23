import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor

import encoder_transformation

original = pd.read_csv('C:/Users/jax/Desktop/pythonProject/v2/preprocessed/original_preprocessed.csv')
additional = pd.read_csv('C:/Users/jax/Desktop/pythonProject/v2/preprocessed/additional_preprocessed.csv')
all = pd.concat([original,additional], axis=0)
all = all.reset_index(drop=True)

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

    # Define the parameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0),
        'early_stopping_rounds': 10
    }
    # Train the model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


# Run the Optuna study
study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Print the best hyperparameters found by Optuna
print("Best hyperparameters: ", study.best_params)

