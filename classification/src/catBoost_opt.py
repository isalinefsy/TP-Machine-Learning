import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Chargement et prétraitement des données
dataTrain = pd.read_csv('../data/train.csv')
if dataTrain.isnull().sum().any():
    numeric_cols = dataTrain.select_dtypes(include=[np.number]).columns
    dataTrain[numeric_cols] = dataTrain[numeric_cols].fillna(dataTrain[numeric_cols].mean())
dataTrain['date'] = pd.to_datetime(dataTrain['date']).astype(str)

features = ['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']
target = 'bc_price_evo'
X = dataTrain[features]
y = dataTrain[target]

cat_features = ['date']
cat_feature_indices = [X.columns.get_loc(col) for col in cat_features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 300, 1500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.1),
        'depth': trial.suggest_int('depth', 6, 12),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 20),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 3),
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostClassifier(**param)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print("  Best hyperparameters: ", trial.params)
