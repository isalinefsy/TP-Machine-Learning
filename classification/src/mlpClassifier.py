import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier  # Importation de MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# Chargement des données (à adapter selon votre dataset)
dataTrain = pd.read_csv('../data/train.csv') 

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

# Split des données en 80% entraînement et 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Entraînement du modèle MLPClassifier sur les données d'entraînement avec GridSearchCV
grid_search = GridSearchCV(MLPClassifier(random_state=42), { 
    'hidden_layer_sizes': [(50,), (100,), (200,)],  # Nombre de neurones dans chaque couche cachée
    'activation': ['tanh', 'relu'],  # Fonction d'activation
    'solver': ['adam', 'sgd'],  # Optimiseur
    'alpha': [0.0001, 0.001],  # Terminaison de régularisation
    'max_iter': [200, 300]  # Nombre d'itérations d'entraînement
}, cv=5, scoring='accuracy', n_jobs=-1)  # Utiliser tous les cœurs du processeur pour accélérer la recherche
grid_search.fit(X_train, y_train)

# Meilleur modèle
clf = grid_search.best_estimator_
print("Meilleurs hyperparamètres : ", grid_search.best_params_)

# Prédictions sur les données de test (10%)
y_pred_test = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy sur les données de test : {accuracy_test * 100:.2f}%")

# Sauvegarde des prédictions sur les données de test dans un fichier de soumission
dataTest = pd.read_csv('../data/test.csv')

# Prétraitement des données de test
for column in dataTest.select_dtypes(include=['object']).columns:
    if column in label_encoders:  # Vérifier si la colonne a été encodée lors de l'entraînement
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
submission.to_csv('../data/target/MLP_submission.csv', index=False)
print("Fichier de soumission '../data/target/MLP_submission.csv' généré avec succès !")
