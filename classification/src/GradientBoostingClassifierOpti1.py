import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
X = dataTrain[features].copy()  # Utilisation de .copy() pour éviter les warnings
y = dataTrain[target]

# Normalisation des données
scaler = StandardScaler()
X.loc[:, features] = scaler.fit_transform(X[features])  # Utilisation de .loc pour éviter les warnings

# Split des données en 85% entraînement et 15% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Hyperparamètres optimaux obtenus lors de l'essai 44
best_params = {
    'n_estimators': 950,
    'learning_rate': 0.1653890843449149,
    'max_depth': 16,
    'min_samples_split': 31,
    'min_samples_leaf': 4,
    'subsample': 0.9,
    'max_features': None,
    'loss': 'exponential'
}

# Entraînement du modèle avec les hyperparamètres optimaux
print("Entraînement du modèle en cours...")
clf_best = GradientBoostingClassifier(random_state=42, **best_params)
clf_best.fit(X_train, y_train)

# Évaluation sur le jeu de validation
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
dataTest.loc[:, features] = scaler.transform(dataTest[features])  # Utilisation de .loc pour éviter les warnings

X_final_test = dataTest[features]
preds = clf_best.predict(X_final_test)

# Conversion des prédictions en "UP" ou "DOWN"
preds_mapped = pd.Series(preds).map({1: 'UP', 0: 'DOWN'})
dataTest['bc_price_evo'] = preds_mapped

# Sauvegarde du fichier de soumission
submission = dataTest[['id', 'bc_price_evo']]
submission.to_csv('../data/target/GradientBoosting_Bayes_submission1.csv', index=False)
print("Fichier de soumission '../data/target/GradientBoosting_Bayes_submission1.csv' généré avec succès !")