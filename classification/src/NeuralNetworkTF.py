# # Classification avec TensorFlow


#Importation des bibliothèques nécessaires
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split



# Chargement des données
df = pd.read_csv("../data/train.csv")

# Encode Up/Down en 0 et 1
df['bc_price_evo'] = df['bc_price_evo'].map({'UP': 1, 'DOWN': 0})

# Séparation des données
X = df.iloc[:, 1:-1].values  # 7 entrées
Y = df.iloc[:, -1].values    # Valeur cible (0, 1)
IDs = df.iloc[:, 0].values  # ID pour suivi

# Séparation des données en ensembles d'entraînement et de validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

print("Données d'entraînement :")
print(X_train)
print(Y_train)
print(IDs)

# Arrêt précoce pour éviter le surapprentissage
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Définition du modèle
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(7,),
                 kernel_initializer='he_normal', bias_initializer='random_normal'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', 
                 kernel_initializer='he_normal', bias_initializer='random_normal'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', 
                 kernel_initializer='he_normal', bias_initializer='random_normal'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu', 
                 kernel_initializer='he_normal', bias_initializer='random_normal'),
    layers.Dense(1, activation='sigmoid',  # Sortie entre 0 et 1
                 kernel_initializer='glorot_uniform', bias_initializer='random_normal')
])


# Compilation du modèle
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

# Entraînement
epochs = 100
model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs,callbacks=[early_stopping], batch_size=32, verbose=2)

# Prédiction sur un nouveau dataset
df_test = pd.read_csv("../data/test.csv")
X_test = df_test.iloc[:, 1:].values
IDs_test = df_test.iloc[:, 0].values

predictions = model.predict(X_test).flatten()  # Prédictions entre 0 et 1

# Conversion des prédictions en UP/DOWN
labels = np.where(predictions >= 0.5, "UP", "DOWN")

# Sauvegarde des résultats
output_df = pd.DataFrame({"id": IDs_test, "bc_price_evo": labels})
output_df.to_csv("../data/NeuralNetworkTF_submission.csv", index=False)

print("Prédictions sauvegardées dans NeuralNetworkTF_submission.csv")


