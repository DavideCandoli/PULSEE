from Operators import Operator, Density_Matrix, Observable, Random_Operator, Commutator
import math
import numpy as np
from scipy import linalg
import hypothesis.strategies as st
from hypothesis import given, note, assume

# Checks that the constructor of the class Operator raises error when it receives an input array which is not a square matrix
def test_Operator_Initialisation_with_Wrong_Dimensions():
    wrong_input = np.ones((1, 2))
    try:
        o = Operator(wrong_input)
        raise AssertionError
    except IndexError:
        pass
    except AssertionError:
        raise AssertionError("No AssertionError caused by the initialisation with a non-square array")
        
# Checks that the constructor of the class Operator raises error when it receives an input array whose elements cannot be cast into complex
def test_Operator_Initialisation_with_Wrong_Matrix_Elements():
    wrong_input = np.array([['a', 'b'], ['c', 'd']])
    try:
        o = Operator(wrong_input)
        raise AssertionError
    except ValueError:
        pass
    except AssertionError:
        raise AssertionError("No ValueError caused by the initialisation through an array with wrong values")
        
# Checks that the constructor of the class Operator raises error when it receives an input scalar whose value cannot be interpreted as an integer
def test_Operator_Initialisation_with_Wrong_Scalar_Value():
    wrong_input = 'a'
    try:
        o = Operator(wrong_input)
        raise AssertionError
    except ValueError:
        pass
    except AssertionError:
        raise AssertionError("No ValueError caused by the initialisation with a string")

# Checks that the constructor of the class Operator raises error when it receives an argument of invalid type (e.g. a list)
def test_Operator_Initialisation_with_Wrong_Argument_Type():
    wrong_input = [1, 'a', "goodbye"]
    try:
        o = Operator(wrong_input)
        raise AssertionError
    except TypeError:
        pass
    except AssertionError:
        raise AssertionError("No ValueError caused by the initialisation with a list")
       
        
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

# Checks that an Operator o to the power of -1 is the reciprocal of o
@given(d = st.integers(min_value=1, max_value=16))
def test_Reciprocal_Operator(d):
    o = Random_Operator(d)
    o_r = o**(-1)
    note("o = %r" % (o.matrix))
    note("o_r = %r" % (o_r.matrix))
    assert np.all(np.isclose((o*o_r).matrix, np.identity(d), rtol=1e-10))
    
# Checks the fact that the eigenvalues of the exponential of an Operator o are the exponentials of
# o's eigenvalues
@given(d = st.integers(min_value=1, max_value=8))
def test_Exponential_Operator_Eigenvalues(d):
    o = Random_Operator(d)
    o_e, o_v = linalg.eig(o.matrix)
    exp_e, exp_v = linalg.eig(o.exp().matrix)
    sorted_exp_o_e = np.sort(np.exp(o_e))
    sorted_exp_e = np.sort(exp_e)
    note("o = %r" % (o.matrix))
    note("exp(o) = %r" % (o.exp().matrix))
    note("Eigenvalues of o = %r" % (np.sort(o_e)))
    note("Exponential of the eigenvalues of o = %r" % (sorted_exp_o_e))
    note("Eigenvalues of exp(o) = %r" % (sorted_exp_e))
    assert np.all(np.isclose(sorted_exp_o_e, sorted_exp_e, rtol=1e-10)) # <--- Not always verified!?!

# Checks that the similarity transformation is equivalent to diagonalising an Operator o when the chosen change of basis operator has the eigenvectors of o as columns
@given(d = st.integers(min_value=1, max_value=8))
def test_Diagonalising_Change_Of_Basis(d):
    o = Random_Operator(d)
    o_e, o_v = linalg.eig(o.matrix)
    p = Operator(o_v)
    o_sim = o.sim_trans(p).matrix
    o_diag = np.diag(o_e)
    note("o = %r" % (o.matrix))
    note("o (diagonalised through the sim_trans method) = %r" % (o_sim))
    note("o (expressed in diagonal form through direct computation of the eigenvalues) = %r"
         % (o_diag))
    note("Eigenvalues of o = %r" % (o_e))
    assert np.all(np.isclose(o_sim, o_diag, rtol=1e-10))

@given(d = st.integers(min_value=1, max_value=16))
def test_Trace_Invariance_Under_Similarity(d):
    o = Random_Operator(d)
    singularity = True
    while(singularity):
        p = Random_Operator(d)
        try:
            o_sim = o.sim_trans(p)
            singularity = False
        except LinAlgError:
            pass
    diff = np.absolute(o.trace()-o_sim.trace())
    note("o = %r" % (o.matrix))
    note("p^{-1}op = %r" % (o_sim.matrix))
    note("Trace of o = %r" % (o.trace()))
    note("Trace of p^{-1}op = %r" % (o_sim.trace()))
    note("diff = %r" % (diff))
    assert diff < 1e-10

# Checks that the adjoint of an Operator o's exponential is the exponential of the adjoint of o
@given(d = st.integers(min_value=1, max_value=16))
def test_Adjoint_Exponential(d):
    o = Random_Operator(d)
    o_exp = o.exp()
    left_hand_side = (o_exp.dagger()).matrix
    right_hand_side = ((o.dagger()).exp()).matrix
    note("(exp(o))+ = %r" % (left_hand_side))
    note("exp(o+) = %r" % (right_hand_side))    
    assert np.all(np.isclose(left_hand_side, right_hand_side, rtol=1e-10))


    