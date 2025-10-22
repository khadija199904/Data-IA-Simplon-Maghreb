import pytest
import pandas as pd
from pipeline import df, df_cleaned, best_mae  

# --- Test : vérifier les colonnes mal typées ---
def test_format():
    erreurs = []
    for col in df.columns:
        if df[col].dtype == "object":   # pandas utilise 'object', pas 'Object'
            try:
                pd.to_numeric(df[col])
                erreurs.append(col)
            except:
                pass
    assert not erreurs, f"Colonnes avec type incorrect : {erreurs}"
    print("Le test de format est réussi : toutes les colonnes numériques ont le bon type.")

# --- Test : vérifier que le dataset nettoyé garde les mêmes dimensions ---
def test_dim():
    assert df.shape == df_cleaned.shape, f"Dimensions différentes : {df.shape} vs {df_cleaned.shape}"
    print("Test des dimensions réussi :", df_cleaned.shape)

# --- Test : vérifier le MAE ---
def test_mae():
    seuil = 10
    assert best_mae <= seuil, f"MAE trop élevé : {best_mae}"
    print("Test de seuil MAE réussi :", best_mae)
