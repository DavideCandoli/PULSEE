import math
import numpy as np

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term

from Nuclear_Spin import Nuclear_Spin

from Hamiltonians import H_Zeeman, H_Quadrupole

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
    
    
    
    
    
    
    
    
    
    
    
    