import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import optuna

# Chargement des données
dataTrain = pd.read_csv('../data/train.csv')  # Adaptez le chemin

# Traitement des valeurs manquantes
if dataTrain.isnull().sum().any():
    print("Attention : des valeurs manquantes ont été détectées. Remplacement par la moyenne.")
    dataTrain.fillna(dataTrain.mean(), inplace=True)

# Encodage des variables catégoriques
label_encoders = {}
for column in dataTrain.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    dataTrain[column] = le.fit_transform(dataTrain[column])
    label_encoders[column] = le

# Définition des features et de la cible
features = ['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']
target = 'bc_price_evo'
X = dataTrain[features]
y = dataTrain[target]

# Split des données en 90% entraînement et 10% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
    }
    clf = GradientBoostingClassifier(random_state=42, **params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    return accuracy_score(y_val, preds)

# Création de l'étude bayésienne avec 10 essais
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Meilleur essai :")
trial = study.best_trial
print("  Accuracy :", trial.value)
print("  Hyperparamètres optimaux :", trial.params)

# Entraînement du modèle final avec les hyperparamètres optimaux sur l'ensemble d'entraînement
clf_best = GradientBoostingClassifier(random_state=42, **trial.params)
clf_best.fit(X_train, y_train)
y_pred_val = clf_best.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f"Accuracy sur le jeu de validation : {accuracy_val * 100:.2f}%")

# Préparation des données de test pour la soumission
dataTest = pd.read_csv('../data/test.csv')
for column in dataTest.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        dataTest[column] = label_encoders[column].transform(dataTest[column])
    else:
        print(f"Avertissement : la colonne {column} n'a pas été encodée lors de l'entraînement.")

X_final_test = dataTest[features]
preds = clf_best.predict(X_final_test)

# Conversion des prédictions en "UP" ou "DOWN"
preds_mapped = pd.Series(preds).map({1: 'UP', 0: 'DOWN'})
dataTest['bc_price_evo'] = preds_mapped

# Sauvegarde du fichier de soumission
submission = dataTest[['id', 'bc_price_evo']]
submission.to_csv('../data/target/GradientBoosting_Bayes_submission.csv', index=False)
print("Fichier de soumission '../data/target/GradientBoosting_Bayes_submission.csv' généré avec succès !")
