# Brief 1 : Prédiction du Churn Client (Désabonnement) – Projet Data Science

## Objectif
L’objectif de ce projet est de développer un pipeline complet de Machine Learning supervisé pour prédire le désabonnement (churn) des clients d’une entreprise de télécommunications.

## Description
Ce projet couvre toutes les étapes d’un cycle complet de Data Science :

Exploration et analyse du dataset (EDA) à l’aide d’un notebook Jupyter.

Préparation et traitement des données dans un script Python structuré.

Entraînement et évaluation de plusieurs modèles de classification.

Mise en place de tests unitaires pour garantir la fiabilité du pipeline.

Comparaison et sélection du meilleur modèle selon des métriques de performance clés.

---
Prédiction-Churn-Client/
│
├── EDA_and_visualisations.ipynb   # Analyse exploratoire des données
├── pipeline.py                    # Préparation des données et entraînement des modèles
├── test_pipeline.py               # Tests unitaires automatisés
├── requirements.txt               # Liste des dépendances du projet
├── README.md                      # Documentation du projet
└── data/                          # Dossier contenant les données sources


## Résultats et choix du modèle
| Modèle                               | Accuracy | Recall | F1-score |
|--------------------------------------|----------|--------|----------|
| Logistic Regression classifier       | 0.8204   | 0.5925 | 0.6360   |
| SVC                                  | 0.8097   | 0.5040 | 0.58     |


Le modèle de Régression Logistique a été retenu car il offre la meilleure F1-score (0.64) et un bon équilibre entre précision et rappel.
Des essais de sélection de caractéristiques (feature selection) ont également été réalisés pour améliorer la performance globale du modèle.

## Environnement et installation

```bash
git clone https://github.com/Data-IA-Simplon-Maghreb/Prédiction-Churn-Client.git
cd Prédiction-Churn-Client
pip install -r requirements.txt
 ```
### Exécution du pipeline
```bash
python pipeline.py
 ```
### Lancer les tests unitaires
```bash
pytest test_pipeline.py

 ```
## Technologies et librairies utilisées

- Python 3.10+

- Pandas, NumPy

- Matplotlib, Seaborn

- Scikit-learn

- Pytest

## Perspectives d’amélioration
- Sélection des features pour améliorer la performance.

- Équilibrage du dataset (ex : SMOTE).

- Tester d’autres modèles comme XGBoost.

- Optimisation des hyperparamètres avec GridSearchCV.

 - Automatisation du pipeline avec MLflow.


