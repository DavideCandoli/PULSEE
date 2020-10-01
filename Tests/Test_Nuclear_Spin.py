import sys
sys.path.insert(1, '../Code')

import math
import numpy as np

import hypothesis.strategies as st
from hypothesis import given, note

from Operators import *
from Nuclear_Spin import *
        

def test_spin_quantum_number_initialisation_with_non_half_integer():
    wrong_input = 2.6
    try:
        I = Nuclear_Spin(wrong_input)
        raise AssertionError
    except ValueError as ve:
        if not "The given spin quantum number is not a half-integer number" in ve.args[0]:
            raise AssertionError("No ValueError caused by the initialisation of the spin quantum number with a non-half-integer number")
    except AssertionError:
        raise AssertionError("No ValueError caused by the initialisation of the spin quantum number with a non-half-integer number")
        
@given(s = st.integers(min_value=1, max_value=14))
def test_spin_raising_lowering_operators_are_hermitian_conjugate(s):
    n_s = Nuclear_Spin(s/2)
    raising_dagger = n_s.I['+'].dagger()
    lowering = n_s.I['-']
    note("Adjoint of I_raising = %r" % (raising_dagger.matrix))
    note("I_lowering = %r" % (lowering.matrix))
    assert np.all(np.isclose(raising_dagger.matrix, lowering.matrix, rtol=1e-10))

# Checks the well-known commutation relation
# [I_x, I_y] = i I_z
# between the cartesian components of the spin
@given(s = st.integers(min_value=1, max_value=14))
def test_spin_commutation_relation(s):
    n_s = Nuclear_Spin(s/2)
    left_hand_side = commutator(n_s.I['x'], n_s.I['y'])
    right_hand_side = 1j*n_s.I['z']
    note("[I_x, I_y] = %r" % (left_hand_side.matrix))
    note("i I_z = %r" % (right_hand_side.matrix))
    assert np.all(np.isclose(left_hand_side.matrix, right_hand_side.matrix, rtol=1e-10))

    
