import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

# Suppresion des warnings
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)


# plotting
import matplotlib.pyplot as plt
import seaborn as sns

"""
with open("train.csv") as my_csv_file:
    header_and_data = list(csv.reader(my_csv_file, delimiter=','))
"""

#Notre objectif va être de trouver une "formule" pour calculer l'émission de CO2
# Ici nous avons un fichier CSV contenant les données d'entrainement
# On va ensuite essayer d'entrainer un modèle de régression linéaire sur ces données

#On va d'abord charger les données
#On va utiliser pandas pour ça

train_data = pd.read_csv("train.csv")

#On va charger les données de test
test_data = pd.read_csv("test.csv")

""" Methode 2 : using csv.reader
header = header_and_data[0] #Name of the columns
data = header_and_data[1:] #Data of the columns
"""

#print(train_data.info())
#print(train_data.isna().sum())

#On constate qu'il y a beaucoup de lignes avec la colonne hc manquante
#Cependant, il y a une colonne hcnox qui est la somme de hc + nox, donc en soit on peut supprimer la colonne hc, puisque hcnox reprend les valeurs

#On remplace la colonne brand par la moyenne de la colonne co2 pour chaque marque
#On va donc créer une nouvelle colonne hc, qui vaudra la moyenne de la colonne co2 pour chaque marque
#On va créer un dictionnaire avec les marques et la moyenne de la colonne co2 pour chaque marque
# Calcul de la moyenne de CO2 par marque à partir du train_data
dictionary = {}
for index, row in train_data.iterrows():
    brand = row["brand"]
    co2 = row["co2"]
    if brand not in dictionary:
        dictionary[brand] = [co2]
    else:
        dictionary[brand].append(co2)

# Calcul des moyennes pour chaque marque
for key in dictionary:
    dictionary[key] = np.mean(dictionary[key])

# Remplacement de la colonne brand par la moyenne de CO2 pour chaque marque dans le jeu d'entraînement
train_data["brand"] = train_data["brand"].map(dictionary)

# Pour le jeu de test, on remplace également brand par la moyenne de CO2 pour la marque
# Si une marque du jeu de test n'a pas été vue dans le train, on utilise la moyenne globale de CO2
global_mean = train_data["co2"].mean()
test_data["brand"] = test_data["brand"].map(dictionary).fillna(global_mean)

# On peut ensuite supprimer la colonne "hc"
train_data = train_data.drop(columns=["hc"])
test_data = test_data.drop(columns=["hc"])


#On va supprimer les lignes avec des valeurs manquantes, ce qui va supprimer environ 9000 lignes sur 41000
#On verra par la suite si c'est trop, et dans ce cas on essayera de faire des approximations sur les valeurs manquantes

#Après tests, on retire trop de données, donc overfitting
#On ne va retirer que les hc, mais on va remplacer les valeurs manquantes par la moyenne de la colonne pour les autres null


numeric_cols_train = train_data.select_dtypes(include=[np.number]).columns
train_data[numeric_cols_train] = train_data[numeric_cols_train].fillna(train_data[numeric_cols_train].mean())

numeric_cols_test = test_data.select_dtypes(include=[np.number]).columns
test_data[numeric_cols_test] = test_data[numeric_cols_test].fillna(train_data[numeric_cols_train].mean())


"""
Ici nous avons essayé de remplacer valeurs manquantes par la médiane plutôt que par la moyenne, mais les résultats sont moins satisfaisants

def replace_missing_with_median(train_data, test_data):
    #On va d'abord créer une fonction qui va remplacer les valeurs manquantes par la médiane de la colonne pour les deux jeux de données
    numeric_cols_train = train_data.select_dtypes(include=[np.number]).columns
    numeric_cols_test = test_data.select_dtypes(include=[np.number]).columns

    #On va remplacer les valeurs manquantes par la médiane de la colonne pour les deux jeux de données
    for col in numeric_cols_train:
        median = train_data[col].median()
        train_data[col] = train_data[col].fillna(median)

    for col in numeric_cols_test:
        median = test_data[col].median()
        test_data[col] = test_data[col].fillna(median)

    return train_data, test_data

replace_missing_with_median(train_data, test_data)
"""

#On supprime les doublons
train_data = train_data.drop_duplicates(keep='first')


"""
#Dans cette partie nous avons essayé de supprimer les données aberrantes grâce à la méthode de l'écart interquartile, mais après test 
#nous avons constaté que cela supprimait trop de données, donc nous avons décidé de ne pas l'utiliser

#On regarde si il y a des données aberrantes, et si oui on les supprime
#On va utiliser la méthode de l'écart interquartile pour ça

# Sélectionner uniquement les colonnes numériques
train_data_numeric = train_data.select_dtypes(include=[np.number])

# Calcul des quartiles
Q1 = train_data_numeric.quantile(0.25)
Q3 = train_data_numeric.quantile(0.75)
IQR = Q3 - Q1

IQR = Q3 - Q1

# Création du masque pour identifier les valeurs dans l'intervalle acceptable
mask = (train_data_numeric >= (Q1 - 7 * IQR)) & (train_data_numeric <= (Q3 + 7 * IQR))

# Appliquer le masque aux colonnes numériques uniquement
train_data_numeric_filtered = train_data_numeric[mask].dropna()

# Conserver les colonnes non numériques et les réintégrer
train_data_filtered = train_data.loc[train_data_numeric_filtered.index]

train_data = train_data_filtered

print(train_data.shape)

"""



#Via des labels:
test_data['co2'] = 0

#Combinaison des données d'entrainement et de test pour l'encodage cohérent
combined_data = pd.concat([train_data, test_data], keys=['train', 'test'])

label_encoder = LabelEncoder()

categorical_columns = ["model", "car_class", "range", "fuel_type", "hybrid", "grbx_type_ratios"]
for col in categorical_columns:
    combined_data[col] = label_encoder.fit_transform(combined_data[col])

#On re sépare les ensembles
train_data = combined_data.loc['train']
test_data = combined_data.loc['test']

X = train_data.drop(columns=["co2",'id'])
y = train_data["co2"]


#On va séparer les données en deux parties : une partie pour l'apprentissage, et une partie pour le test
#On va utiliser 80% des données pour l'apprentissage, et 20% pour le test
#On va utiliser la fonction train_test_split de sklearn pour ça

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



"""
#Autre méthode : le One Hot Encoding --> moins performant que le Label Encoding d'après nos tests


#train_data = pd.get_dummies(train_data, columns=["model","brand", "car_class", "range", "fuel_type", "hybrid", "grbx_type_ratios"], drop_first=True)


#test_data = pd.get_dummies(test_data, columns=["model","brand", "car_class", "range", "fuel_type", "hybrid", "grbx_type_ratios"], drop_first=True)
#Le hot One Encoding ne marche pas par la suite car les données de test ne contiennent pas toutes les colonnes du train_data

#On va donc donc concaténer les données de test et d'entrainement, puis on va faire le One Hot Encoding sur l'ensemble des données
#Puis on les resépare après (on va donc ajouter un attribut pour pouvoir les reséparer après)

# On fait une copie de train_data et test_data pour éviter les modifications indésirables
train_data_enc = train_data.copy()
test_data_enc = test_data.copy()

#Je dois ajouter une colonne "co2" dans le test_data, qui vaudra 0, pour pouvoir faire le One Hot Encoding
# On va créer une colonne "co2" dans test_data_enc avec des valeurs nulles
test_data_enc["co2"] = np.nan

# Ajout d'un indicateur pour pouvoir les reséparer après
train_data_enc["is_train"] = True
test_data_enc["is_train"] = False

# Dans test_data_enc, il n'y a pas de colonne 'co2'
# On garde la colonne "id" dans test_data_enc, qui servira pour l'export final

# Concaténation des deux datasets
combined_data = pd.concat([train_data_enc, test_data_enc], sort=False)

# On applique le One-Hot Encoding sur l'ensemble des données
combined_data = pd.get_dummies(
    combined_data,
    columns=["model", "brand", "car_class", "range", "fuel_type", "hybrid", "grbx_type_ratios"],
    drop_first=True
)

# Séparation des données reconstituées
train_data_final = combined_data[combined_data["is_train"] == True].drop(columns=["is_train"])
test_data_final = combined_data[combined_data["is_train"] == False].drop(columns=["is_train"])

# On s'assure de retirer la colonne "co2" du test, si elle existe
test_data_final = test_data_final.drop(columns=["co2"], errors='ignore')
"""

#Maintenant que nous avons nettoyer les données, essayons de faire un apprentissage pour déterminer les valeurs de CO2

#On essaie une regression linéaire
#On va essayer de prédire la valeur de CO2 en fonction des autres colonnes


"""
#On va maintenant entrainer le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained")
print(f"Score : {model.score(X_test, y_test)}")

#Le Score affiché est le R2, qui est un indicateur de la qualité du modèle


#Je vais calculer la mean absolute error, qui est une autre mesure de la qualité du modèle
mae = np.mean(np.abs(model.predict(X_test) - y_test))
print(f"Mean absolute error : {mae}")
"""

#On va essayer un autre modèle, le Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

"""
# Définition des hyperparamètres à tester
rf_params = {
    'n_estimators': [150, 200],  # Nombre d'arbres
    'max_depth': [20, None],  # Profondeur max
    'min_samples_split': [2, 5],  # Nombre min d'échantillons pour diviser un nœud
    'min_samples_leaf': [1, 2, 4],  # Nombre min d'échantillons dans un nœud terminal
}

# Recherche des meilleurs hyperparamètres en minimisant MAE
rf_model = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring='r2',  # MAE négatif (pour minimiser MAE)
    n_jobs=-1,
    verbose=3  # Afficher les détails de la recherche
)

"""
"""
#Sans paramètre
rf_model = RandomForestRegressor(random_state=42, n_estimators=200)

rf_model.fit(X_train, y_train)

# Affichage des résultats
#print("Meilleurs paramètres :", rf_model.best_params_)
#print(f"Meilleur MAE (valeur absolue) : {-rf_model.best_score_}")

print("Random Forest Model trained")
print(f"Score : {rf_model.score(X_test, y_test)}")
print(f"Mean absolute error : {np.mean(np.abs(rf_model.predict(X_test) - y_test))}")
"""

#On va essayer un autre modèle, le Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

"""
gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), {
    'n_estimators': [50, 100, 150],  # Nombre d'arbres (estimators)
    'learning_rate': [0.01, 0.05, 0.1],  # Taux d'apprentissage
    'max_depth': [3, 5, 10],  # Profondeur des arbres
    'min_samples_split': [2, 5],  # Nombre minimum d'échantillons pour diviser un nœud
    'min_samples_leaf': [1, 2]  # Nombre minimum d'échantillons dans un nœud terminal
}, cv=5, scoring='r2', n_jobs=-1, verbose=3)  # Utiliser tous les cœurs du processeur pour accélérer la recherche


#Sans hyperparamètres
gb_model = GradientBoostingRegressor(random_state=42, n_estimators=200)

gb_model.fit(X_train, y_train)
print("Gradient Boosting Model trained")
print(f"Score : {gb_model.score(X_test, y_test)}")
print(f"Mean absolute error : {np.mean(np.abs(gb_model.predict(X_test) - y_test))}")
"""
import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error


#On va essayer un autre modèle, le XGBoost Regressor
#XGBoost est une implémentation optimisée de Gradient Boosting

#On va utiliser la méthode de GridSearchCV pour trouver les meilleurs hyperparamètres

# Recherche des meilleurs hyperparamètres pour XGBRegressor

"""
param_grid = {
    'n_estimators': [2000],  # Nombre d'arbres (estimators)
    'learning_rate': [0.05, 0.1],  # Taux d'apprentissage
    'max_depth': [3, 5, 7],  # Profondeur des arbres
    'min_child_weight': [1, 5, 10],  # Poids minimal d'un enfant dans un nœud
    'subsample': [0.8, 0.9, 1.0],  # Fraction des données utilisées à chaque itération
    'colsample_bytree': [0.7, 0.8],  # Fraction des colonnes utilisées pour chaque arbre
    'gamma': [0, 0.1],  # Réduction de la perte requise pour diviser un nœud
    'scale_pos_weight': [1, 10],  # Pesée des classes déséquilibrées (si applicable)
}

# GridSearchCV pour trouver les meilleurs hyperparamètres
xgb_model = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',  # Utilisation du R2 pour évaluer la performance
    n_jobs=-1,  # Utilisation de tous les cœurs pour accélérer la recherche
    verbose=3  # Afficher les étapes du GridSearch
)
"""

#sans hyperparamètres
xgb_model = XGBRegressor(n_estimators=2000,random_state=42)

xgb_model.fit(X_train, y_train)
print("XGBoost Model trained")
print(f"Score : {xgb_model.score(X_test, y_test)}")
print(f"Mean absolute error : {np.mean(np.abs(xgb_model.predict(X_test) - y_test))}")



# Affichage des meilleurs hyperparamètres
#print("Meilleurs hyperparamètres : ", random_search.best_params_)

# Utilisation du meilleur modèle pour faire des prédictions
#best_model = random_search.best_estimator_

# Évaluation du modèle sur les données de test
#print(f"Score : {xgb_model.score(X_test, y_test)}")
#print(f"Mean absolute error : {np.mean(np.abs(xgb_model.predict(X_test) - y_test))}")


#Dans l'ordre, on constate que le meilleur modèle pour nos données est XGBRegression, puis random Forest, Gradient Boosting, et enfin la LinearRegression
#On va donc utiliser le XGBRegression pour faire des prédictions sur les données de test



#Je dois produire un document csv, avec un header contenant "id" et "co2", et les valeurs de co2 prédites par le modèle


#On va maintenant faire des prédictions sur les données de test

data_test = test_data.drop(columns=["id","co2"])

#predictions = rf_model.predict(data_test)

#predictions = gb_model.predict(data_test)

predictions = xgb_model.predict(data_test)

#On va maintenant créer un csv avec les id et les valeurs prédites
#On va utiliser la fonction to_csv de pandas pour ça
#On va créer un dataframe avec les id et les valeurs prédites
#On va utiliser la fonction pd.DataFrame pour ça

predictions_df = pd.DataFrame({"id": test_data["id"], "co2": predictions})
predictions_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")

