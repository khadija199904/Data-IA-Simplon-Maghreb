import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler ,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import  class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)

from test_pipeline import test_shape_ligne


# ----------------------------------------------------------
# Fonction pour charger dataset 
# ----------------------------------------------------------

def load_dataset (filepath) :
    # Charger le fichier CSV
    df = pd.read_csv(filepath)
    print(" Dataset chargé avec success :", df.shape)
    print(df.head())
    return df 
# ----------------------------------------------------------
# fonction pour nettoyer la dataset 
# ----------------------------------------------------------

def cleaning_data(df,cols_inutile= None):
    """
    Nettoie le DataFrame en effectuant les opérations suivantes :
    - Suppression des doublons
    - Suppression des colonnes inutiles
    - Conversion de 'TotalCharges' en numérique
    - Remplissage des valeurs manquantes (numériques et catégorielles)
    """
    df_cleaned = df.copy()
    # suprimmer les colonnes inutiles 
    if cols_inutile :
        df_cleaned = df_cleaned.drop(columns= cols_inutile, errors='ignore')

    # Convertir TotalCharges en numérique
    if 'TotalCharges' in df_cleaned.columns:
        df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], errors='coerce')

    # handling missing values
    df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    # Remove the dupilcate row
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned
    
# ----------------------------------------------------------
# Function for encoding the categorial columns
# ----------------------------------------------------------

def encode_categorical(df, binary_cols=None, ordinal_cols=None):

    """
    Encode les colonnes catégorielles :
    - Colonnes binaires et Colonnes ordinales → LabelEncoder
    - Autres colonnes → One-Hot Encoding

    Retourne :
        df_encoded : DataFrame encodée
    """

    df_encoded = df.copy()
    le = LabelEncoder()
    
    
    # Fusionner les colonnes binaires et ordinales
    binary_oridinal_cols = binary_cols + ordinal_cols
    
    # Encoder toutes ces colonnes avec LabelEncoder
    for col in binary_oridinal_cols:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # Encodage One-Hot pour les autres
    other_catg_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    if len(other_catg_cols) > 0:
        df_encoded = pd.get_dummies(df_encoded, columns=other_catg_cols,drop_first=True, dtype=int)

    return df_encoded

# -----------------------------
# Fonction dr normalisation
# -----------------------------

def normalize_data(df_numeric, method):
    """
    Normalise les colonnes numériques d'un DataFrame .
    Méthode de normalisation :
      - 'minmax' : applique MinMaxScaler (par défaut)
      - 'standard' : applique StandardScaler
    
    """

    df_scaled = df_numeric.copy()
    # Détecter les colonnes booléennes dans les numériques
    bool_cols = []
    numer_cols = []
    for col in df_scaled.columns:
        unique_vals = df_scaled[col].dropna().unique()
        if len(unique_vals) > 0 and set(unique_vals).issubset({0, 1}):
            bool_cols.append(col)
        else:
            numer_cols.append(col)
    # Choisir le scaler
    if method == 'minmax':
        scaler = MinMaxScaler()
        
    elif method == 'standard':
        scaler = StandardScaler()
        
    else:
        raise ValueError("❌ Méthode non reconnue. Choisir parmi: 'minmax', 'standard'")
    # Appliquer la normalisation uniquement sur les colonnes numériques
    if numer_cols:
        df_scaled[numer_cols] = scaler.fit_transform(df_scaled[numer_cols])
    
    
    return df_scaled
# -----------------------------
# Fonction Split data 
# -----------------------------

def split_data (df,target_col) :

   X = df.drop(columns=[target_col])
   y = df[target_col]

   X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

   return X_train, X_test, y_train, y_test
# -----------------------------
# Fonction d'entraînement
# -----------------------------

def traning_model ( X_train, y_train, model_name):
   
    if  model_name == 'LogisticRegression':
       # Create an instance of LogisticRegression classifier
          model = LogisticRegression()
       # Create an instance of Support victor classifier 
    elif model_name == "SVC":
        model = SVC(probability=True)  # probability=True pour ROC/PR   
       # Create an instance of Random Forest classifier
    elif model_name == 'RFC' :
        model = RandomForestClassifier()
    else:
        raise ValueError("Model name not recognized. Choose 'LogisticRegression' or 'SVC'.")
    
    # Fit the model
    model.fit(X_train, y_train)

    return model

# -----------------------------
# Fonction d'évaluation
# -----------------------------
def evaluate_model(model,X_test,y_test,labels=None): 
    """
    Évalue un modèle de classification sur des données de test.
    
    
    Returns:
        rapport de classification 
        Accuracy, Recall, F1-score, courbe ROC ,PR_Curve et confusion Matrix
        
        """
    
    # Create the predictions

    y_predict = model.predict(X_test)
 
    # Calcul des métriques globales
    metrics = {

    "accuracy": accuracy_score(y_test, y_predict),
    "recall": recall_score(y_test, y_predict),
    "f1": f1_score(y_test, y_predict)
               }
    # Convertir en DataFrame pour tableau
    metrics_table = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])


    #  Matrice de confusion
    cm = confusion_matrix(y_test, y_predict)
    
    # -----------------------------------------------
    # Probabilités pour ROC/PR
    # ----------------------------------------------- 
    # La plupart des modèles comme LogisticRegression ou RandomForest
    # ont la méthode predict_proba() qui retourne un tableau [n_samples, n_classes]
    # Probabilités pour ROC et PR
    # Certains modèles comme SVC (sans probability=True) n'ont pas predict_proba
    # mais ont decision_function() qui retourne un "score" indiquant
    # la distance à la frontière de décision
    # On peut utiliser ce score pour tracer ROC/PR

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else :
        y_proba = model.decision_function(X_test)
    
    
    # ROC
    # Données pour courbes ROC et PR (pour binaire)
    roc_data = None
    pr_data = None
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_data = (fpr, tpr, roc_auc)
    
    # Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    pr_data = (recall_vals, precision_vals, pr_auc)
    

    return metrics, cm, roc_data, pr_data
# -----------------------------
# Sauvegarde
# -----------------------------

def save_evaluation_resultes(results,save_dir="results") : 
    """
    Sauvegarde métriques, matrice de confusion et courbes ROC/PR pour un modèle.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Tableau comparatif pour tous les modèles
    comparison_table = pd.DataFrame()

    for model_name, (metrics_table, cm, roc_data, pr_data) in results.items():
        model_subdir = os.path.join(save_dir, model_name)
        os.makedirs(model_subdir, exist_ok=True)
        
        # Sauvegarde tableau des métriques
        
        df_metrics = pd.DataFrame.from_dict(metrics_table, orient='index', columns=['Score'])

        # Ajouter au tableau comparatif
        temp = df_metrics.copy()
        temp.columns = [model_name]  # nom du modèle en colonne
        comparison_table = pd.concat([comparison_table, temp], axis=1)

        # Matrice de confusion
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Matrice de confusion")
        plt.xlabel("Prédictions")
        plt.ylabel("Valeurs réelles")
        plt.savefig(os.path.join(model_subdir, "confusion_matrix.png"))
        plt.close()

        # Courbes ROC 
        if roc_data is not None:
            fpr, tpr, roc_auc = roc_data
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
            plt.plot([0,1],[0,1],'k--')
            plt.title(f"ROC Curve - {model_name}")
            plt.xlabel('Faux Positifs')
            plt.ylabel('Vrais Positifs')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(model_subdir, "roc_curve.png"))
            plt.close()

        if pr_data is not None:
            recall_vals, precision_vals, pr_auc = pr_data
            plt.figure()
            plt.plot(recall_vals, precision_vals, label=f'PR (AUC = {pr_auc:.2f})')
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(os.path.join(model_subdir, "pr_curve.png"))
            plt.close()
    # Sauvegarder le tableau comparatif global
    comparison_table.to_csv(os.path.join(save_dir, "comparison_metrics.csv"))
    print(f"✅ Tous les résultats sauvegardés dans {save_dir}")

    return

def main():
    
    print("\ Démarrage du pipeline...\n")
    df = load_dataset("C:\Users\khadija\Data-IA-Simplon-Maghreb\Prédiction-Churn-Client\Churn_client-dataset.csv")  
    

    df_cleaned = cleaning_data(df,cols_inutile = ['customerID','gender'])

    df_encoded = encode_categorical(df_cleaned,binary_cols=['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'], ordinal_cols=['Contract'])
    df_encoded.head()

    df_scaled = normalize_data(df_encoded,method='minmax')

    target_col = "Churn"  
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = split_data(df_scaled, target_col)

    # Vérification shapes
    test_shape_ligne(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
# ---------------------------------------------------------------------------------------------- 
# Entraînement 
# ---------------------------------------------------------------------------------------------- 
    model_name = ["LogisticRegression", "SVC"]
    results = {}
    for m in model_name:
      model = traning_model (X_train_scaled,y_train_scaled, m)
      metrics_table, cm, roc_data, pr_data, = evaluate_model(model, X_test, y_test)
      results[m] = (metrics_table, cm, roc_data, pr_data)
    
    save_evaluation_resultes(results,save_dir="resultas de l'evaluation ")
# -----------------------------
# Lancement
# -----------------------------
if __name__ == "__main__":
    main()
       
       
    

