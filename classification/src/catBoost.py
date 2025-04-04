import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Chargement des données d'entraînement
dataTrain = pd.read_csv('../data/train.csv')

# Traitement des valeurs manquantes pour les colonnes numériques
if dataTrain.isnull().sum().any():
    numeric_cols = dataTrain.select_dtypes(include=[np.number]).columns
    dataTrain[numeric_cols] = dataTrain[numeric_cols].fillna(dataTrain[numeric_cols].mean())

# Conversion de 'date' en string pour que CatBoost le traite comme catégoriel
dataTrain['date'] = pd.to_datetime(dataTrain['date']).astype(str)

# Définition des features et de la cible
features = ['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']
target = 'bc_price_evo'

X = dataTrain[features]
y = dataTrain[target]

# Pour CatBoost, on utilise la gestion native des variables catégoriques.
# Ici, on considère que 'date' est catégorielle.
cat_features = ['date']
cat_feature_indices = [X.columns.get_loc(col) for col in cat_features]

# Séparation des données en 90% entraînement et 10% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

# Initialisation du modèle CatBoostClassifier avec les hyperparamètres optimaux
model = CatBoostClassifier(
    bagging_temperature=0,
    depth=10,
    iterations=300,
    l2_leaf_reg=5,
    learning_rate=0.1,
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

# Entraînement du modèle
model.fit(train_pool, eval_set=val_pool)

# Prédiction sur le jeu de validation et calcul de l'accuracy
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"✅ Accuracy CatBoost : {acc * 100:.2f}%")

# Préparation du jeu de test pour la soumission
dataTest = pd.read_csv('../data/test.csv')

# Traitement des valeurs manquantes dans le jeu de test
if dataTest.isnull().sum().any():
    numeric_cols_test = dataTest.select_dtypes(include=[np.number]).columns
    dataTest[numeric_cols_test] = dataTest[numeric_cols_test].fillna(dataTest[numeric_cols_test].mean())

# Conversion de 'date' en string dans le jeu de test
dataTest['date'] = pd.to_datetime(dataTest['date']).astype(str)

X_final = dataTest[features]

# Prédiction sur le jeu de test
final_preds = model.predict(X_final)
final_preds = pd.Series(final_preds)

# Mapping des prédictions (si elles sont numériques)
if np.issubdtype(final_preds.dtype, np.number):
    final_preds_mapped = final_preds.map({1: 'UP', 0: 'DOWN'})
else:
    final_preds_mapped = final_preds

dataTest['bc_price_evo'] = final_preds_mapped

# Sauvegarde du fichier de soumission avec le bon format
submission = dataTest[['id', 'bc_price_evo']]
submission.to_csv('../data/target/CatBoost_GridSearch_submission.csv', index=False)
print("📁 Fichier de soumission '../data/target/CatBoost_GridSearch_submission.csv' généré avec succès !")
