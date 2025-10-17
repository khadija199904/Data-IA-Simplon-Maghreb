import pytest



def test_shape(df,df_cleaned) :
    
    assert df.shape == df_cleaned.shape , "dimension de df n'est le meme que le df netoyyé "

    

def test_mae(mae):
    seuil = 10 
    assert mae <= seuil