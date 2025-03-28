import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

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

#On va donc supprimer la colonne hc
train_data = train_data.drop(columns=["hc","brand"])
test_data = test_data.drop(columns=["hc","brand"])


#On va supprimer les lignes avec des valeurs manquantes, ce qui va supprimer environ 9000 lignes sur 41000
#On verra par la suite si c'est trop, et dans ce cas on essayera de faire des approximations sur les valeurs manquantes

#Après tests, on retire trop de données, donc overfitting
#On ne va retirer que les hc, mais on va remplacer les valeurs manquantes par la moyenne de la colonne pour les autres null

numeric_cols_train = train_data.select_dtypes(include=[np.number]).columns
train_data[numeric_cols_train] = train_data[numeric_cols_train].fillna(train_data[numeric_cols_train].mean())

numeric_cols_test = test_data.select_dtypes(include=[np.number]).columns
test_data[numeric_cols_test] = test_data[numeric_cols_test].fillna(train_data[numeric_cols_train].mean())

#On supprime les doublons
train_data = train_data.drop_duplicates(keep='first')

#print(train_data.shape)

"""

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


#Maintenant que nous avons nettoyer les données, essayons de faire un apprentissage pour déterminer les valeurs de CO2

#On essaie une regression linéaire
#On va essayer de prédire la valeur de CO2 en fonction des autres colonnes
#On va d'abord séparer les données en deux parties : une partie pour l'apprentissage, et une partie pour le test
#On va utiliser 80% des données pour l'apprentissage, et 20% pour le test
#On va utiliser la fonction train_test_split de sklearn pour ça
"""

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Random Forest Model trained")
print(f"Score : {rf_model.score(X_test, y_test)}")
print(f"Mean absolute error : {np.mean(np.abs(rf_model.predict(X_test) - y_test))}")

"""
#On va essayer un autre modèle, le Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
print("Gradient Boosting Model trained")
print(f"Score : {gb_model.score(X_test, y_test)}")
print(f"Mean absolute error : {np.mean(np.abs(gb_model.predict(X_test) - y_test))}")"
"""

#Dans l'ordre, on constate que le meilleur modèle pour nos données est random Forest, puis le Gradient Boosting, puis la régression linéaire
#On va donc utiliser le Random Forest pour faire des prédictions sur les données de test



#Je dois produire un document csv, avec un header contenant "id" et "co2", et les valeurs de co2 prédites par le modèle
#Je vais donc créer un dataframe avec les données de test, et les valeurs prédites par le modèle



"""""
#Afficher les colonnes différentes entre test_data et train_data

# Obtenir les colonnes des deux datasets
train_columns = set(X_train.columns)
test_columns = set(test_data.columns)

# Colonnes présentes dans train_data mais absentes dans test_data
missing_in_test = train_columns - test_columns
print("Colonnes manquantes dans test_data :", missing_in_test)

# Colonnes présentes dans test_data mais absentes dans train_data
extra_in_test = test_columns - train_columns
print("Colonnes en trop dans test_data :", extra_in_test)
"""

#On va maintenant faire des prédictions sur les données de test

data_test = test_data.drop(columns=["id","co2"])

predictions = rf_model.predict(data_test)


#On va maintenant créer un csv avec les id et les valeurs prédites
#On va utiliser la fonction to_csv de pandas pour ça
#On va créer un dataframe avec les id et les valeurs prédites
#On va utiliser la fonction pd.DataFrame pour ça

predictions_df = pd.DataFrame({"id": test_data["id"], "co2": predictions})
predictions_df.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")
