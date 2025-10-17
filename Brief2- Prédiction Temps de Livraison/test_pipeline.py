import pytest

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

def test_dim(df,df_cleaned) :
    
    assert df.shape == df_cleaned.shape 
    print(" Test des dimensions réussi ",df_cleaned.shape)

    

def test_mae(mae):
    seuil = 10 
    assert mae <= seuil
    print(" Test de seuil mae réussi ")