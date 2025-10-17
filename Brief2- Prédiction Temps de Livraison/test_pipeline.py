import pytest

def test_dim(df,df_cleaned) :
    
    assert df.shape == df_cleaned.shape 
    print(" Test des dimensions réussi ",df_cleaned.shape)

    

def test_mae(mae):
    seuil = 10 
    assert mae <= seuil
    print(" Test de seuil mae réussi ")