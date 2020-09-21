import math
import numpy as np

from Operators import Operator, Density_Matrix, Observable, \
                      random_density_matrix, \
                      random_operator

from Many_Body import tensor_product_operator, partial_trace

import hypothesis.strategies as st
from hypothesis import given, note


@given(d = st.integers(min_value=1, max_value=8))
def test_tensor_product_conserves_density_matrix_properties(d):
    A = random_density_matrix(d)
    B = random_density_matrix(d)
    
    try:
        C = tensor_product_operator(A, B)
    except ValueError as ve:
        if "The input array lacks the following properties: \n" in ve.args[0]:
            error_message = ve.args[0][49:]
            error_message = "The tensor product of two Density_Matrix objects lacks the following properties: \n" + error_message
            note("A = %r" % (A.matrix))
            note("B = %r" % (B.matrix))
            raise AssertionError(error_message)

            
@given(d = st.integers(min_value=2, max_value=6))
def test_partial_trace_is_inverse_tensor_product(d):
    A = random_operator(d-1)
    A = A/A.trace()
    B = random_operator(d)
    B = B/B.trace()
    C = random_operator(d+1)
    C = C/C.trace()
    
    AB = tensor_product_operator(A, B)
    BC = tensor_product_operator(B, C)
    ABC = tensor_product_operator(AB, C)
    
    p_t = partial_trace(ABC, [d-1, d, d+1], 0)
    
    assert np.all(np.isclose(p_t.matrix, BC.matrix, rtol=1e-10))

