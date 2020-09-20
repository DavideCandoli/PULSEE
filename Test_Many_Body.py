import math
import numpy as np

from Operators import Operator, Density_Matrix, Observable, \
                      random_density_matrix, \
                      random_operator

from Many_Body import Tensor_Product_Operator, Partial_Trace

import hypothesis.strategies as st
from hypothesis import given, note

# Checks that the operation implemented by function Tensor_Product_Operator conserves the defining
# properties of density matrices, i.e. hermitianity, unit trace and positivity
@given(d = st.integers(min_value=1, max_value=8))
def test_Tensor_Product_Conserves_Density_Matrix_Properties(d):
    A = random_density_matrix(d)
    B = random_density_matrix(d)
    
    try:
        C = Tensor_Product_Operator(A, B)
    except ValueError as ve:
        if "The input array lacks the following properties: \n" in ve.args[0]:
            error_message = ve.args[0][49:]
            error_message = "The tensor product of two Density_Matrix objects lacks the following properties: \n" + error_message
            note("A = %r" % (A.matrix))
            note("B = %r" % (B.matrix))
            raise AssertionError(error_message)

# Checks that the operation implemented by the function Partial_Trace is the inverse of the one performed
# by Tensor_Product_Operator when they act on unit trace operators
@given(d = st.integers(min_value=2, max_value=6))
def test_Partial_Trace_Inverse_Tensor_Product(d):
    A = random_operator(d-1)
    A = A/A.trace()
    B = random_operator(d)
    B = B/B.trace()
    C = random_operator(d+1)
    C = C/C.trace()
    
    AB = Tensor_Product_Operator(A, B)
    BC = Tensor_Product_Operator(B, C)
    ABC = Tensor_Product_Operator(AB, C)
    
    partial_trace = Partial_Trace(ABC, [d-1, d, d+1], 0)
    
    assert np.all(np.isclose(partial_trace.matrix, BC.matrix, rtol=1e-10))

