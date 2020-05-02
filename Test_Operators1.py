import math
from Operators import Operator, Density_Matrix, Observable, Commutator, Generate_Random_Operator
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, note

@given(d = st.integers(min_value=1, max_value=16))
def test_Operator_Product_Associativity(d):
    a = Generate_Random_Operator(d)
    b = Generate_Random_Operator(d)
    c = Generate_Random_Operator(d)
    left_product = (a*b)*c
    right_product = a*(b*c)
    note("a = %r" % (a.matrix))
    note("b = %r" % (b.matrix))
    note("c = %r" % (c.matrix))
    note("(a*b)*c = %r" % (left_product.matrix))
    note("a*(b*c) = %r" % (right_product.matrix))
    assert np.all(np.isclose(left_product.matrix, right_product.matrix, rtol=1e-10))

@given(d = st.integers(min_value=1, max_value=16))
def test_Trace_Invariance_under_Similarity(d):
    singularity = True
    while(singularity==True):
        p = Generate_Random_Operator(d)
        rank_p = np.linalg.matrix_rank(p.matrix, tol=1e-10)
        if rank_p == d: singularity=False
    o = Generate_Random_Operator(d)
    o_sim = o.sim_trans(p)
    note("o = %r" % (o.matrix))
    note("o = %r" % (o.matrix))
    assert math.isclose(o.trace(), o_sim.trace(), rtol=1e-10)
    
# Checks the validity of the formula for the commutation of the exponential
def test_Operator_Exponential_Commutation():
    A = Operator(np.array([[],[]]))
        B = Random_Operator(d)
        C = Commutator(A, B)
        condition_1 = np.all(np.isclose(Commutator(C, A).matrix, np.zeros(d)))
        condition_2 = np.all(np.isclose(Commutator(C, B).matrix, np.zeros(d)))
    left_hand_side = B.exp()*A.exp()
    right_hand_side = A.exp()*B.exp()*(-C).exp()
    assert np.all(np.isclose(left_hand_side, right_hand_side, rtol=1e-10))


    