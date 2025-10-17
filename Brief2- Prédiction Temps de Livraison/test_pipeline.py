import pytest



def test_dim(df,df_cleaned) :
    
    assert df.shape == df_cleaned.shape 
    print(" Test des dimensions réussi ")

    

def test_mae(mae):
    seuil = 10 
    assert mae <= seuil