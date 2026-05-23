# ❤️ Prédiction des Maladies Cardiovasculaires (CardioPredict)

Ce projet vise à détecter le risque de maladies cardiovasculaires à l'aide de modèles de Machine Learning (Classification), en partant d'une analyse approfondie des données cliniques jusqu'au déploiement d'une application web interactive.

---

## 🛠️ Étapes du Projet

Le projet a été développé de manière séquentielle, documenté étape par étape dans notre notebook Jupyter, avant d'être déployé.

### 1. Analyse Uni-variable

- **Statistiques descriptives :** Calcul des mesures de tendance centrale et de dispersion pour chaque variable.
- **Nettoyage des données :**
  - Gestion des valeurs manquantes.
  - Identification et traitement des valeurs aberrantes (Outliers).
- **Visualisations :** Représentation graphique des distributions de chaque variable.

### 2. Analyse Bi-variables

- **Entre variables explicatives :** Calcul et interprétation de la Matrice de Corrélation de Pearson pour détecter la colinéarité.
- **Entre variables explicatives et la cible (Y) :** Analyse de l'impact individuel de chaque variable (âge, IMC, pression artérielle, etc.) sur la présence ou l'absence d'une maladie cardiovasculaire.

### 3. Analyse Multi-variable

- Étude combinée des relations complexes entre plusieurs paramètres cliniques et anthropométriques.

### 4. Préparation des Données d'Entraînement

- **Création de variables synthétiques :** Calcul de l'IMC (BMI) et de la Pression Artérielle Moyenne (PAM/MAP).
- **Séparation Features/Target (X et Y) :** Isolement de la variable cible.
- **Split Train/Test :** Séparation des données pour l'apprentissage et l'évaluation.
- **Transformations :**
  - Transformation de Box-Cox pour normaliser la distribution des données asymétriques (exportation des `lambda_values.pkl`).
  - Standardisation (Z-score) via un `StandardScaler` (exportation de `scaler.pkl`).

### 5. Modélisation Machine Learning

Plusieurs modèles de classification ont été entraînés et comparés :

1. **Régression Logistique**
2. **Random Forest**
3. **XGBoost**

- **Évaluation :** Utilisation des métriques de précision (Accuracy) et de l'AUC-ROC.
- **Sélection du meilleur modèle :** **XGBoost** a été retenu pour ses performances supérieures (~72% de précision et excellente AUC).
- Le modèle final a été exporté sous le format `model_xgboost.pkl`.

### 6. Interface Web Interactive (Streamlit)

Pour rendre le modèle accessible, une interface web "CardioPredict" a été développée avec **Streamlit**.

- L'utilisateur renseigne les paramètres cliniques d'un patient.
- Le backend applique **exactement le même pipeline de préparation** (Box-Cox, StandardScaler).
- Le modèle **XGBoost** génère une probabilité et un verdict (Risque Élevé / Profil Favorable).

---

## 🚀 Comment lancer l'application locale ?

1. **Assurez-vous d'avoir Python 3.9+ installé.**
2. **Installez les dépendances nécessaires** :
   ```bash
   pip install -r requirements.txt
   ```
3. **Lancement de l'interface Streamlit** :
   ```bash
   streamlit run app.py
   ```
4. Ouvrez votre navigateur sur l'adresse indiquée (généralement `http://localhost:8501`).
