from Operators import Operator, Density_Matrix, Observable, Random_Operator
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, note

# Checks that the constructor of the class Operator raises error when it receives an input array which is not a square matrix
def test_Operator_Initialisation_with_Wrong_Dimensions():
    wrong_input = np.ones((1, 2))
    try:
        o = Operator(wrong_input)
        raise Exception("No AssertionError caused by the initialisation with a non-square array")
    except AssertionError:
        pass
    except Exception as noerror:
        assert 1==0, print(noerror)
        
# Checks that the constructor of the class Operator raises error when it receives an input array whose elements cannot be cast into complex
def test_Operator_Initialisation_with_Wrong_Values():
    wrong_input = np.array([['a', 'b'], ['c', 'd']])
    try:
        o = Operator(wrong_input)
        raise Exception("No ValueError caused by the initialisation through an array with wrong values")
    except ValueError:
        pass
    except Exception as noerror:
        assert 1==0, print(noerror)

# Checks that the difference between identical operators returns a null square array
@given(d = st.integers(min_value=1, max_value=16))
def Opposite_Operator(d):
    o = Random_Operator(d)
    note("o = %r" % (o.matrix))
    assert np.all(np.isclose((o-o).matrix, np.zeros((d,d)), rtol=1e-10))

# Checks that the sum of operators is associative
@given(d = st.integers(min_value=1, max_value=16))
def test_Operator_Sum_Associativity(d):
    a = Random_Operator(d)
    b = Random_Operator(d)
    c = Random_Operator(d)
    left_sum = (a+b)+c
    right_sum = a+(b+c)
    note("a = %r" % (a.matrix))
    note("b = %r" % (b.matrix))
    note("c = %r" % (c.matrix))
    note("(a+b)+c = %r" % (left_sum.matrix))
    note("a+(b+c) = %r" % (right_sum.matrix))
    assert np.all(np.isclose(left_sum.matrix, right_sum.matrix, rtol=1e-10))

# Checks that the product of operators is associative
@given(d = st.integers(min_value=1, max_value=16))
def test_Operator_Product_Associativity(d):
    a = Random_Operator(d)
    b = Random_Operator(d)
    c = Random_Operator(d)
    left_product = (a*b)*c
    right_product = a*(b*c)
    note("a = %r" % (a.matrix))
    note("b = %r" % (b.matrix))
    note("c = %r" % (c.matrix))
    note("(a*b)*c = %r" % (left_product.matrix))
    note("a*(b*c) = %r" % (right_product.matrix))
    assert np.all(np.isclose(left_product.matrix, right_product.matrix, rtol=1e-10))

# Checks that the distributive property is valid with the current definition of operators + and *
@given(d = st.integers(min_value=1, max_value=16))
def test_Operator_Distributivity(d):
    a = Random_Operator(d)
    b = Random_Operator(d)
    c = Random_Operator(d)
    left_hand_side = a*(b+c)
    right_hand_side = a*b+a*c
    note("a = %r" % (a.matrix))
    note("b = %r" % (b.matrix))
    note("c = %r" % (c.matrix))
    note("(a*(b+c) = %r" % (left_hand_side.matrix))
    note("a*b+a*c = %r" % (right_hand_side.matrix))
    assert np.all(np.isclose(left_hand_side.matrix, right_hand_side.matrix, rtol=1e-10))


