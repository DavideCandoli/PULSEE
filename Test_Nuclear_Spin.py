import math
import numpy as np

import hypothesis.strategies as st
from hypothesis import given, note

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
        
# Checks that the raising and lowering operators are hermitian conjugate
@given(s = st.integers(min_value=1, max_value=14))
def test_Nuclear_Spin_Raising_Lowering_Hermitian_Conjugate(s):
    n_s = Nuclear_Spin(s/2)
    raising_dagger = n_s.I['+'].dagger()
    lowering = n_s.I['-']
    note("Adjoint of I_raising = %r" % (raising_dagger.matrix))
    note("I_lowering = %r" % (lowering.matrix))
    assert np.all(np.isclose(raising_dagger.matrix, lowering.matrix, rtol=1e-10))
    
# Checks the well-known commutation relation
# [I_x, I_y] = i hbar I_z
# between the cartesian components of the spin (for computational purposes, we have set hbar=1)
@given(s = st.integers(min_value=1, max_value=14))
def test_Nuclear_Spin_Commutation_Relation(s):
    n_s = Nuclear_Spin(s/2)
    left_hand_side = Commutator(n_s.I['x'], n_s.I['y'])
    right_hand_side = 1j*n_s.I['z']
    note("[I_x, I_y] = %r" % (left_hand_side.matrix))
    note("i I_z = %r" % (right_hand_side.matrix))
    assert np.all(np.isclose(left_hand_side.matrix, right_hand_side.matrix, rtol=1e-10))    

    
