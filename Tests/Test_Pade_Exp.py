import sys
sys.path.insert(1, '../Code')

import math
import numpy as np
from scipy import linalg
from scipy.linalg import eig

import hypothesis.strategies as st
from hypothesis import given, settings, note, assume

from Operators import random_operator
from Pade_Exp import expm

@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_convergence_of_pade_approximant(d):
    r_o = random_operator(d)
    r_m = r_o.matrix
    
    exp_r_m_5 = expm(r_m, q=5)
    exp_r_m_10 = expm(r_m, q=10)
    
    assert np.all(np.isclose(exp_r_m_5, exp_r_m_10, rtol=1e-10))