import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Chargement des données
dataTrain = pd.read_csv('../data/train.csv')

# Vérification et traitement des valeurs manquantes
if dataTrain.isnull().sum().any():
    print("⚠️ Valeurs manquantes détectées. Remplacement par la moyenne.")
    dataTrain.fillna(dataTrain.mean(numeric_only=True), inplace=True)

# Encodage des colonnes catégoriques
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

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Création du modèle XGBoost avec des paramètres bien choisis (similaires à ceux trouvés par grid search)
xgb_clf = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    eval_metric='logloss'
)

# Entraînement
xgb_clf.fit(X_train, y_train)

# Prédiction et évaluation
y_pred_test = xgb_clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"✅ Accuracy sur les données de test : {accuracy_test * 100:.2f}%")

# Chargement des données de test
dataTest = pd.read_csv('../data/test.csv')

# Encodage des colonnes catégoriques
for column in dataTest.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        dataTest[column] = label_encoders[column].transform(dataTest[column])
    else:
        print(f"⚠️ Avertissement : colonne {column} non vue à l'entraînement.")

X_final_test = dataTest[features]

# Prédictions
dataTest['bc_price_evo'] = xgb_clf.predict(X_final_test)
dataTest['bc_price_evo'] = dataTest['bc_price_evo'].map({1: 'UP', 0: 'DOWN'})

# Création du fichier de soumission
submission = dataTest[['id', 'bc_price_evo']]
submission.to_csv('../data/XGBoost_submission.csv', index=False)
print("📁 Fichier de soumission '../data/XGBoost_submission.csv' généré avec succès.")
