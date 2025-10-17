# import labriries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.preprocessing import LabelEncoder ,StandardScaler 
from test_Pipeline import test_dim ,test_format




# Cleaning dataset
def cleaning_data(df):
    """
    Nettoie le DataFrame en effectuant les opérations suivantes :
    - Suppression des doublons
    - Conversion de 'TotalCharges' en numérique
    - Remplissage des valeurs manquantes (numériques et catégorielles)
    """
    df_cleaned = df.copy()
    
    # Remove the dupilcate row
    df_cleaned = df_cleaned.drop_duplicates()
    # Check missing value 
    df.isna().sum()
    # fill missing value for numérique variable 
    df_cleaned["Courier_Experience_yrs"] = df_cleaned["Courier_Experience_yrs"] .fillna(df_cleaned["Courier_Experience_yrs"] .mean())
    # fill missing value for catégoriale variable 
    catg_cols = df_cleaned.select_dtypes(include='object').columns
    for col in catg_cols :
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0]) # la valeur le plus fréquentes 
    
    
    return df_cleaned




# **************** Main ************************

if __name__ == "__main__":

 df = pd.read_csv(r"C:\Users\khadija\Desktop\simplon project\Data-IA-Simplon-Maghreb\Brief2- Prédiction Temps de Livraison\Prédiction-Temps-Livraison.csv")
 print(df.head())

 df_cleaned = cleaning_data(df)
 

 test_dim(df,df_cleaned)

 test_format(df)