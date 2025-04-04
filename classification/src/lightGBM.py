import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import early_stopping, log_evaluation


# 📥 Chargement des données
dataTrain = pd.read_csv('../data/train.csv')

# 🧼 Nettoyage
if dataTrain.isnull().sum().any():
    dataTrain.fillna(dataTrain.mean(numeric_only=True), inplace=True)

# 🔠 Encodage des colonnes object
label_encoders = {}
for col in dataTrain.select_dtypes(include='object').columns:
    le = LabelEncoder()
    dataTrain[col] = le.fit_transform(dataTrain[col])
    label_encoders[col] = le

# 🧪 Feature engineering utile
dataTrain['price_diff'] = dataTrain['bc_price'] - dataTrain['ab_price']
dataTrain['demand_diff'] = dataTrain['bc_demand'] - dataTrain['ab_demand']
dataTrain['is_peak_hour'] = dataTrain['hour'].apply(lambda h: 1 if 7 <= h <= 22 else 0)

features = [
    'date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer',
    'price_diff', 'demand_diff', 'is_peak_hour'
]
target = 'bc_price_evo'

X = dataTrain[features]
y = dataTrain[target]

# 🔀 Split avec early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ⚙️ Paramètres du modèle
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=10,
    num_leaves=50,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.4,
    random_state=42
)

# 🧠 Entraînement avec early stopping

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='binary_error',
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]
)


# 📈 Prédiction sur la validation
y_pred_val = lgb_model.predict(X_val)
acc = accuracy_score(y_val, y_pred_val)
print(f"✅ Accuracy LightGBM : {acc * 100:.4f}%")

# 📤 Prédictions sur le test final
dataTest = pd.read_csv('../data/test.csv')
for col in dataTest.select_dtypes(include='object').columns:
    if col in label_encoders:
        dataTest[col] = label_encoders[col].transform(dataTest[col])

dataTest['price_diff'] = dataTest['bc_price'] - dataTest['ab_price']
dataTest['demand_diff'] = dataTest['bc_demand'] - dataTest['ab_demand']
dataTest['is_peak_hour'] = dataTest['hour'].apply(lambda h: 1 if 7 <= h <= 22 else 0)

X_final = dataTest[features]
dataTest['bc_price_evo'] = lgb_model.predict(X_final)
dataTest['bc_price_evo'] = dataTest['bc_price_evo'].map({1: 'UP', 0: 'DOWN'})

# 💾 Sauvegarde
submission = dataTest[['id', 'bc_price_evo']]
submission.to_csv('../data/LightGBM_submission.csv', index=False)
print("📁 Fichier '../data/LightGBM_submission.csv' généré avec succès.")
