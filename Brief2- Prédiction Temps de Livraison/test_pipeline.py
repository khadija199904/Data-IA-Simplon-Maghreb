import pytest

def test_shape(df,df_cleaned) :
    
    assert df.shape[0] == df_cleaned.shape[0] 

    assert df.shape[1] == df_cleaned.shape[1]

def test_mae(mae):
    seuil = 10 
    assert mae <= seuil