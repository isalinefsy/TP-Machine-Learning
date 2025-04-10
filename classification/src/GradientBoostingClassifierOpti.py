import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Normalisation des données
scaler = StandardScaler()
X[features] = scaler.fit_transform(X[features])

# Split des données en 85% entraînement et 15% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

def objective(trial):
    # Ajout de nouveaux hyperparamètres à optimiser
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'loss': trial.suggest_categorical('loss', ['deviance', 'exponential'])
    }
    clf = GradientBoostingClassifier(random_state=42, **params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    return accuracy_score(y_val, preds)

# Augmentation du nombre d'essais pour Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Augmenté à 100 essais

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

# Normalisation des données de test
dataTest[features] = scaler.transform(dataTest[features])

X_final_test = dataTest[features]
preds = clf_best.predict(X_final_test)

# Conversion des prédictions en "UP" ou "DOWN"
preds_mapped = pd.Series(preds).map({1: 'UP', 0: 'DOWN'})
dataTest['bc_price_evo'] = preds_mapped

# Sauvegarde du fichier de soumission
submission = dataTest[['id', 'bc_price_evo']]
submission.to_csv('../data/target/GradientBoosting_Bayes_submission.csv', index=False)
print("Fichier de soumission '../data/target/GradientBoosting_Bayes_submission.csv' généré avec succès !")