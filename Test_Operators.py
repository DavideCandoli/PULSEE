from Operators import Operator, Density_Matrix, Observable
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, note

def test_Operator_Initialisation_with_Wrong_Dimensions():
    wrong_input = np.ones((1, 2))
    try:
        o = Operator(wrong_input)
        raise Exception("No AssertionError caused by the initialisation with a non-square array")
    except AssertionError:
        pass
    except Exception as noerror:
        assert 1==0, print(noerror)
        
def test_Operator_Initialisation_with_Wrong_Values():
    wrong_input = np.array([['a', 'b'], ['c', 'd']])
    try:
        o = Operator(wrong_input)
        raise Exception("No ValueError caused by the initialisation with wrong values in the array")
    except ValueError:
        pass
    except Exception as noerror:
        assert 1==0, print(noerror)