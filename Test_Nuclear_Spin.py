import math
import numpy as np

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term

from Nuclear_Spin import Nuclear_Spin

# Checks that the constructor of the class Nuclear_Spin raises an appropriate error when a string is
# passed as an argument
def test_Nuclear_Spin_Initialisation_with_Wrong_Type():
    wrong_input = 'a'
    try:
        I = Nuclear_Spin(wrong_input)
        raise AssertionError
    except TypeError:
        pass
    except AssertionError:
        raise AssertionError("No TypeError caused by the initialisation with a string")
        
# Checks that the constructor of the class Nuclear_Spin raises an appropriate error when a non-half-
# integer argument is passed
def test_Nuclear_Spin_Initialisation_with_Non_Half_Integer():
    wrong_input = 2.6
    try:
        I = Nuclear_Spin(wrong_input)
        raise AssertionError
    except ValueError:
        pass
    except AssertionError:
        raise AssertionError("No ValueError caused by the initialisation with a non-half-integer number")