import math
import numpy as np

from Operators import Operator, Density_Matrix, Observable, \
                      Random_Density_Matrix

from Many_Body import Tensor_Product_Operator

import hypothesis.strategies as st
from hypothesis import given, assume

@given(d = st.integers(min_value=1, max_value=8))
def test_Tensor_Product_Conserves_Density_Matrix_Properties(d):
    A = Random_Density_Matrix(d)
    B = Random_Density_Matrix(d)
    
    try:
        C = Tensor_Product_Operator(A, B)
    except ValueError as ve:
        if "The input array lacks the following properties: \n" in ve.args[0]:
            error_message = ve.args[0][49:]
            error_message = "The tensor product of two Density_Matrix objects lacks the following properties: \n" + error_message
            note("A = %r" % (A.matrix))
            note("B = %r" % (B.matrix))
            raise AssertionError(error_message)