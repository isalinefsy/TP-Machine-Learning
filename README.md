# ⚡ TP - Machine Learning

Binary classification problem on electricity price

## 🧱 Installation

Clonez le dépôt :

```bash
git clone <https://github.com/isalinefsy/TP-Machine-Learning>

cd TP-Machine-Learning

```

Créez un environnement virtuel (recommandé) :

```bash
python -m venv venv
source venv/bin/activate # Sous Windows : venv\Scripts\activate
```

Installez les dépendances nécessaires à la classification :

```bash
pip install -r classification/requirements.txt
```

🚀 Utilisation

#### 🔹 Classification

Exécutez le script correspondant au modèle que vous souhaitez utiliser.
Par exemple, pour entraîner un modèle avec LightGBM :

```bash
python classification/lightgbm_classifier.py
```

Les fichiers de soumission seront générés dans :

```bash
classification/data/target/
```

Exemples de fichiers :

```bash
MLP_submission.csv
StackingSoft_submission.csv
```

#### 🔹 Régression

Exécutez le script de régression souhaité.
Par exemple :

```bash
python regression/xgb_regressor.py
```

Les résultats des prédictions seront également sauvegardés dans :

```bash
classification/data/target/
```

👥 Auteurs : Hexanôme 32

- Isaline Foissey
- Quentin Mariat
- Pierrick Brossat
- Adrian Abi Saleh
- Jose Southerland Silva
- Gabriel Marchi Mekari
- Billy Villeroy
