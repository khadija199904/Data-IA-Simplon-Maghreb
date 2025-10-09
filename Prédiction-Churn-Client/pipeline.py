import pandas as pd
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Fonction pour charger dataset 
# ----------------------------------------------------------

def load_dataset (filepath) :
    # Charger le fichier CSV
    df = pd.read_csv(filepath)
    print(df.head())
    return df 
# ----------------------------------------------------------
# 
# ----------------------------------------------------------

def supprime_separe_data(df,cols_inutile= None):
    """
    Supprime les colonnes spécifiées et sépare les données en numériques et catégorielles.
    returns
    numeric_data : Sous-ensemble contenant uniquement les colonnes numériques.
    categorical_data : Sous-ensemble contenant uniquement les colonnes catégorielles.
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
    
    # Séparer les colonnes catégorielles et numériques
    df_numer = df_cleaned.select_dtypes(include=['int64', 'float64'])
    df_catg = df_cleaned.select_dtypes(include=['object', 'category'])
          
    return df_cleaned, df_numer, df_catg
    
# ----------------------------------------------------------
# Function for encoding the categorial columns
# ----------------------------------------------------------

def encode_categorical(df, binary_cols=None, ordinal_cols=None):

    """
    Encode les colonnes catégorielles :
    - Colonnes binaires (Yes/No) → 1/0
    - Colonnes ordinales → LabelEncoder
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

def normalize_fetures(df_numeric, method):
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

def traning_model ( X_train, y_train, model):
   
    if  model == 'LogisticRegression':
       # Create an instance of LogisticRegression classifier
          model = LogisticRegression()
       # Create an instance of Support victor classifier 
    elif model == "SVC":
        model = SVC(probability=True)  # probability=True pour ROC/PR   
       # Create an instance of Random Forest classifier
    elif model == 'RFC' :
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
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)

    print(f"=== Classification Metrics du {model} ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Rapport de classification du {model}")
    print(classification_report(y_test, y_predict))

    #  Matrice de confusion
    cm = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    plt.show()
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
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    return 


   
       
       
    







