import pytest


def test_shape_ligne (X_train, X_test, y_train, y_test) :
    """ 
    Check Dimensions cohérentes entre X et y après split
    """
    assert X_train.shape[0] == y_train.shape[0] 

    assert X_test.shape[0] == y_test.shape[0] 