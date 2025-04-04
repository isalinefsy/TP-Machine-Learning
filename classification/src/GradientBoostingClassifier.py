import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# Chargement des données
dataTrain = pd.read_csv('../data/train.csv')  # Adaptez le chemin

# Vérification et traitement des valeurs manquantes
if dataTrain.isnull().sum().any():
    print("Attention : des valeurs manquantes ont été détectées. Elles seront remplacées par la moyenne.")
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

# Split des données en 90% entraînement et 10% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Définition d'une grille d'hyperparamètres (réduite pour accélérer le processus)
param_grid = {
    'n_estimators': [150, 200, 250],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0],
    'max_features': ['log2']
}

# Recherche par GridSearchCV
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Meilleur modèle trouvé
clf = grid_search.best_estimator_
print("Meilleurs hyperparamètres trouvés :", grid_search.best_params_)

# Évaluation sur le jeu de test
y_pred_test = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy sur les données de test : {accuracy_test * 100:.2f}%")

# Préparation des données de test pour la soumission
dataTest = pd.read_csv('../data/test.csv')
for column in dataTest.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        dataTest[column] = label_encoders[column].transform(dataTest[column])
    else:
        print(f"Avertissement : la colonne {column} n'a pas été encodée lors de l'entraînement.")

X_final_test = dataTest[features]

# Prédiction sur les données de test
dataTest['bc_price_evo'] = clf.predict(X_final_test)

# Conversion des prédictions en "UP" ou "DOWN"
dataTest['bc_price_evo'] = dataTest['bc_price_evo'].map({1: 'UP', 0: 'DOWN'})

# Sauvegarde du fichier de soumission
submission = dataTest[['id', 'bc_price_evo']]
submission.to_csv('../data/target/GradientBoosting_optimized_submission_grid.csv', index=False)
print("Fichier de soumission '../data/target/GradientBoosting_optimized_submission_grid.csv' généré avec succès !")
