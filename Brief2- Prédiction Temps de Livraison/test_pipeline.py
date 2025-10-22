import pytest
import pandas as pd
from pipeline import df, df_cleaned, best_mae  
# Test unitaire :  vérifier les colonnes mal typées 
def test_format(df):
    erreurs = []
    for col in df.columns:
        if df[col].dtype == "Object" :
            try:
                pd.to_numeric(df[col])
                erreurs.append(col)
            except  :
                pass
    assert not erreurs ,f"Colonnes avec type incorrect : {erreurs}"
    print("Le test de format est réussi : toutes les colonnes numériques ont le bon type.")
    
# Test unitaire : vérifier que le dataset nettoyé garde les mêmes dimensions
def test_dim(df,df_cleaned) :
    
    assert df.shape == df_cleaned.shape 
    print(" Test des dimensions réussi ",df_cleaned.shape)


def test_mae():
    seuil = 10 
    assert best_mae <= seuil
    print(" Test de seuil mae réussi ")



