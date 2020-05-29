import math
import numpy as np

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term

from Nuclear_Spin import Nuclear_Spin

from Hamiltonians import H_Zeeman, H_Quadrupole, \
                         V0, V1, V2, \
                         H_Single_Mode_Pulse

import hypothesis.strategies as st
from hypothesis import given, note

# Checks that the Observable object returned by the method H_Zeeman changes sign when the angular
# coordinates of the magnetic field undergo the following change
# theta -> pi - theta    phi -> phi + pi
@given(par = st.lists(st.floats(min_value=0, max_value=20), min_size=3, max_size=3))
def test_Zeeman_Hamiltonian_Flipped_Magnetic_Field(par):
    spin = Nuclear_Spin()
    h_z1 = H_Zeeman(spin, par[0], par[1], par[2])
    h_z2 = H_Zeeman(spin, math.pi-par[0], par[1]+math.pi, par[2])
    note("H_Zeeman(theta, phi) = %r" % (h_z1.matrix))
    note("H_Zeeman(pi-theta, phi+pi) = %r" % (h_z2.matrix))
    note("H_Zeeman(pi-theta, phi+pi)+H_Zeeman(theta, phi) = %r" % (np.absolute(h_z1.matrix+h_z2.matrix)))
    assert np.all(np.absolute(h_z1.matrix+h_z2.matrix) < 1e-10)
    
# Checks that the object returned by the method H_Quadrupole is independent of the Euler angle gamma when
# the asymmetry parameter eta=0
@given(gamma = st.lists(st.floats(min_value=0, max_value=2*math.pi), min_size=2, max_size=2))
def test_Symmetrical_EFG(gamma):
    spin = Nuclear_Spin()
    h_q1 = H_Quadrupole(spin, 1, 1, 0, 1, 1, gamma[0])
    h_q2 = H_Quadrupole(spin, 1, 1, 0, 1, 1, gamma[1])
    note("H_Quadrupole(gamma1) = %r" % (h_q1.matrix))
    note("H_Quadrupole(gamma2) = %r" % (h_q2.matrix))
    assert np.all(np.absolute(h_q1.matrix-h_q2.matrix) < 1e-10)
    
# Checks that the formula for V^0 reduces to the half of the parameter eq when the Euler angles are set
# to 0
@given(eq = st.floats(min_value=0, max_value=20))
def test_V0_Reduces_To_eq(eq):
    v0 = V0(eq, 5., 0, 0, 0)
    assert math.isclose(eq/2, v0, rel_tol=1e-10)
    
# Checks that the formula for V^{+/-1} reduces to 0 when the Euler angles are set to 0
def test_V1_Reduces_To_0():
    for sign in [-1, +1]:
        v1 = V1(sign, 5., 5., 0, 0, 0)
        assert np.absolute(v1) < 1e-10
        
# Checks that the formula for V^{+/-2} reduces to
# (eq/(2*sqrt(6)))*eta
# when the Euler angles are set to 0
@given(eta = st.floats(min_value=0, max_value=1))
def test_V2_Reduces_To_eta(eta):
    for sign in [-2, +2]:
        v2 = V2(sign, 5., eta, 0, 0, 0)
        assert np.isclose(v2, 5*eta/(2*math.sqrt(6)), rtol=1e-10)
        
# Checks that the Hamiltonians returned by H_Single_Mode_Pulse at times which differ by an integer multiple of the
# period of the electromagnetic wave is the same
@given(n = st.integers(min_value=-20, max_value=20))
def test_Periodical_Pulse_Hamiltonian(n):
    spin = Nuclear_Spin(1., 1.)
    omega = 5.
    t1 = 1.
    t2 = t1 + n*(2*math.pi)/omega
    h_p1 = H_Single_Mode_Pulse(spin, omega, 10., 0, math.pi/2, 0, t1)
    h_p2 = H_Single_Mode_Pulse(spin, omega, 10., 0, math.pi/2, 0, t2)
    note("H_Single_Mode_Pulse(t1) = %r" % (h_p1.matrix))
    note("H_Single_Mode_Pulse(t2) = %r" % (h_p2.matrix))
    assert np.all(np.isclose(h_p1.matrix, h_p2.matrix, rtol=1e-10))
    
    
    
    
    
    
    
    
    
    
    
    