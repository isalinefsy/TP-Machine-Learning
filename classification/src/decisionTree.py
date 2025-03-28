import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Chargement des données (à adapter selon votre dataset)
# Remplacez cette ligne par le chargement réel de vos données
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

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et optimisation du modèle DecisionTreeClassifier
param_grid = {
    'max_depth': [3, 5, 10, None], # Profondeur maximale de l'arbre
    'min_samples_split': [2, 5, 10], # Nombre minimum d'échantillons requis pour diviser un nœud
    'min_samples_leaf': [1, 2, 4] # Nombre minimum d'échantillons requis pour être à une feuille
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Meilleur modèle
clf = grid_search.best_estimator_
print("Meilleurs hyperparamètres : ", grid_search.best_params_)

# Évaluation sur les données de test
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Courbe ROC (pour un problème binaire uniquement)
if len(np.unique(y)) == 2:
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
else:
    print("La courbe ROC n'est pas applicable pour un problème multi-classes.")

# Pourcentage de prédictions correctes
correct_predictions = np.mean(y_pred == y_test)
print(f"Pourcentage de prédictions correctes : {correct_predictions * 100:.2f}%")