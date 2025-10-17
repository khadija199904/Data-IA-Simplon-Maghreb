# Prédiction du Temps Total de Livraison 

## Objectif du Projet
L’objectif de ce projet est de **prédire le temps total de livraison** d’une commande, depuis la prise de commande jusqu’à la réception par le client.  
Cette prédiction permet à l’entreprise de :

- **Anticiper les retards** de livraison  
- **Informer les clients en temps réel** sur le statut de leur commande  
- **Optimiser l’organisation des tournées** pour améliorer l’efficacité opérationnelle  

---

## Étapes du Projet

1. **Analyse et Exploration des Données (EDA)**  
2. **Prétraitement des Données**  
3. **Sélection de Features**  
4. **Modélisation classique**  
  - Sélection features avec **SelectKBest**  
  - Entraînement les modèles de régression ( Random Forest regressor et support vector regressor)  
  - Optimisation des hyperparamètres via **GridSearchCV**
3. **Approche : Pipeline sklearn complet** 
   - Création d’un **pipeline sklearn** combinant prétraitement, sélection de features (**SelectKBest**) et modèle de régression.  
   - Optimisation des hyperparamètres avec **GridSearchCV**.
5. **Tests Automatisées**  
   - Tests automatisés pour vérifier le format et les dimensions des données  
   - Vérification que la **MAE** maximale ne dépasse pas un seuil défini  

## Résultats 

- Un modèle capable de prédire le temps total de livraison avec une **erreur moyenne acceptable (MAE)**  
- Des dashboards ou rapports simples pour suivre la performance du modèle  
- Une base solide pour intégrer le modèle dans un **système d’information** ou une application de suivi des commandes  

---

## Instructions pour Exécuter le Projet

1. Cloner le dépôt :  
```bash
git clone <URL_DU_DEPOT>
cd projet-livraison
