import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# Chargement des données (à adapter selon votre dataset)
dataTrain = pd.read_csv('../data/train.csv')  # Remplacez par le chemin réel

# Vérification des données manquantes
if dataTrain.isnull().sum().any():
    print("Attention : des valeurs manquantes ont été détectées. Elles seront remplacées par la moyenne.")
    dataTrain.fillna(dataTrain.mean(), inplace=True)

# Encodage des variables catégoriques si nécessaire
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

# Entraînement du modèle sur 100% des données
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Meilleur modèle
clf = grid_search.best_estimator_
print("Meilleurs hyperparamètres : ", grid_search.best_params_)

# Prédictions sur les données d'entraînement
y_pred_train = clf.predict(X)
accuracy_train = accuracy_score(y, y_pred_train)
print(f"Pourcentage de bonnes réponses sur l'ensemble des données d'entraînement : {accuracy_train * 100:.2f}%")

# Chargement des données de test
dataTest = pd.read_csv('../data/test.csv')

# Prétraitement des données de test
for column in dataTest.select_dtypes(include=['object']).columns:
    if column in label_encoders:  # Vérifier si la colonne a été encodée dans l'entraînement
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
submission.to_csv('../data/target/decisionTree_submission.csv', index=False)
print("Fichier de soumission '../data/target/decisionTree_submission.csv' généré avec succès !")
