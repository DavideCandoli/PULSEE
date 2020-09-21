import math
import numpy as np
import pandas as pd

from Operators import *

from Nuclear_Spin import *

from Hamiltonians import *

import hypothesis.strategies as st
from hypothesis import given, note


@given(par = st.lists(st.floats(min_value=0, max_value=20), min_size=3, max_size=3))
def test_zeeman_hamiltonian_changes_sign_when_magnetic_field_is_flipped(par):
    spin = Nuclear_Spin()
    h_z1 = h_zeeman(spin, par[0], par[1], par[2])
    h_z2 = h_zeeman(spin, math.pi-par[0], par[1]+math.pi, par[2])
    note("h_zeeman(theta, phi) = %r" % (h_z1.matrix))
    note("h_zeeman(pi-theta, phi+pi) = %r" % (h_z2.matrix))
    note("h_zeeman(pi-theta, phi+pi)+h_zeeman(theta, phi) = %r" % (np.absolute(h_z1.matrix+h_z2.matrix)))
    assert np.all(np.absolute(h_z1.matrix+h_z2.matrix) < 1e-10)
    
@given(gamma = st.lists(st.floats(min_value=0, max_value=2*math.pi), min_size=2, max_size=2))
def test_h_quadrupole_independent_of_gamma_when_EFG_is_symmetric(gamma):
    spin = Nuclear_Spin()
    h_q1 = h_quadrupole(spin, 1, 0, 1, 1, gamma[0])
    h_q2 = h_quadrupole(spin, 1, 0, 1, 1, gamma[1])
    note("h_quadrupole(gamma1) = %r" % (h_q1.matrix))
    note("h_quadrupole(gamma2) = %r" % (h_q2.matrix))
    assert np.all(np.absolute(h_q1.matrix-h_q2.matrix) < 1e-10)
    
@given(eta = st.floats(min_value=0, max_value=1))
def test_v0_reduces_to_one_half_when_angles_are_0(eta):
    v0 = v0_EFG(eta, 0, 0, 0)
    assert math.isclose(1/2, v0, rel_tol=1e-10)
    
def test_v1_reduces_to_0_when_angles_are_0():
    for sign in [-1, +1]:
        v1 = v1_EFG(sign, 0.5, 0, 0, 0)
        assert np.absolute(v1) < 1e-10
        
@given(eta = st.floats(min_value=0, max_value=1))
def test_v2_becomes_proportional_to_eta_when_angles_are_0(eta):
    for sign in [-2, +2]:
        v2 = v2_EFG(sign, eta, 0, 0, 0)
        assert np.isclose(v2, eta/(2*math.sqrt(6)), rtol=1e-10)
        
# Checks that the Hamiltonians returned by H_Single_Mode_Pulse at times which differ by an integer
# multiple of the period of the electromagnetic wave is the same
@given(n = st.integers(min_value=-20, max_value=20))
def test_Periodical_Pulse_Hamiltonian(n):
    spin = Nuclear_Spin(1., 1.)
    nu = 5.
    t1 = 1.
    t2 = t1 + n/nu
    h_p1 = H_Single_Mode_Pulse(spin, nu, 10., 0, math.pi/2, 0, t1)
    h_p2 = H_Single_Mode_Pulse(spin, nu, 10., 0, math.pi/2, 0, t2)
    note("H_Single_Mode_Pulse(t1) = %r" % (h_p1.matrix))
    note("H_Single_Mode_Pulse(t2) = %r" % (h_p2.matrix))
    assert np.all(np.isclose(h_p1.matrix, h_p2.matrix, rtol=1e-10))
    
# Checks that the superposition of two orthogonal pulses with the same frequency and a phase difference
# of pi/2 is equivalent to the time-reversed superposition of the two same pulses with one of them
# changed by sign
@given(t = st.floats(min_value=0, max_value=20))
def test_Time_Reversal_Equivalent_Opposite_Circular_Polarization(t):
    spin = Nuclear_Spin(1., 1.)
    mode_forward = pd.DataFrame([(5., 10., 0., 0., 0.),
                                 (5., 10., math.pi/2, math.pi/2, 0.)], 
                                columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    mode_backward = pd.DataFrame([(5., 10., 0., 0., 0.),
                                  (5., 10., -math.pi/2, math.pi/2, 0.)], 
                                 columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    h_p_forward = H_Multiple_Mode_Pulse(spin, mode_forward, t)
    h_p_backward = H_Multiple_Mode_Pulse(spin, mode_backward, -t)
    assert np.all(np.isclose(h_p_forward.matrix, h_p_backward.matrix, rtol=1e-10))
    
# Checks that the Hamiltonian of the pulse expressed in a new picture is equal to that in the
# Schroedinger picture when this latter commutes with the operator for the change of picture
def test_Invariant_IP_Pulse_Hamiltonian_When_Commutation_Holds():
    spin = Nuclear_Spin(1., 1.)
    mode = pd.DataFrame([(5., 10., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    h_change_of_picture = 5.*spin.I['x']
    h_pulse = H_Multiple_Mode_Pulse(spin, mode, 10.)
    h_pulse_ip = H_Changed_Picture(spin, mode, h_change_of_picture, h_change_of_picture, 10.)
    assert np.all(np.isclose(h_pulse.matrix, h_pulse_ip.matrix, rtol=1e-10))

    
    
    
    
    
    