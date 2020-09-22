import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import xticks, yticks

import hypothesis.strategies as st
from hypothesis import given, assume

from Operators import *

from Nuclear_Spin import *

from Hamiltonians import *

from Simulation import *


def test_null_zeeman_contribution_for_0_gyromagnetic_ratio():
    spin_par = {'quantum number' : 3/2,
                'gamma/2pi' : 0.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    h_unperturbed = nuclear_system_setup(spin_par, zeem_par, quad_par)[1]
    
    null_matrix = np.zeros((4, 4))
    
    assert np.all(np.isclose(h_unperturbed.matrix, null_matrix, rtol=1e-10))
    

def test_pi_pulse_yields_population_inversion():
    spin_par = {'quantum number' : 5/2,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    initial_state = np.zeros((6, 6))
    initial_state[0, 0] = 1
    
    spin, h_unperturbed, dm_initial = nuclear_system_setup(spin_par, zeem_par, quad_par, \
                                                           initial_state=initial_state)
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_initial, \
                        mode, pulse_time=1, \
                        picture='IP')
        
    assert np.all(np.isclose(dm_evolved.matrix[5, 5], 1, rtol=1e-1))
    

def test_RRF_operator_proportional_to_Iz_for_theta_0():
    
    spin = Nuclear_Spin(3/2, 1.)
    
    RRF_par = {'nu_RRF': 10,
              'theta_RRF': 0,
              'phi_RRF': 0}
    
    RRF_o = RRF_operator(spin, RRF_par)
    
    RRF_matrix = RRF_o.matrix
    Iz_matrix = spin.I['z'].matrix
    
    c = RRF_matrix[0, 0]/Iz_matrix[0, 0]
    
    assert np.all(np.isclose(RRF_matrix, c*Iz_matrix, rtol=1e-10))
    
    
@given(s = st.integers(min_value=1, max_value=14))
def test_correct_number_lines_power_absorption_spectrum(s):
    
    spin_par = {'quantum number' : s/2,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : math.pi/4,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 5.,
                'asymmetry parameter' : 0.3,
                'alpha_q' : math.pi/3,
                'beta_q' : math.pi/5,
                'gamma_q' : 0}
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, zeem_par, quad_par)
    
    f, p = power_absorption_spectrum(spin, h_unperturbed, normalized=False, dm_initial=dm_0)
    
    assert len(f)==(spin.d)*(spin.d-1)/2
    
# Checks that for very short relaxation times, the FID signal goes rapidly to 0
def test_Fast_Decay_FID_Signal():
    spin_par = {'quantum number' : 2,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : math.pi/4,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 5.,
                'asymmetry parameter' : 0.3,
                'alpha_q' : math.pi/3,
                'beta_q' : math.pi/5,
                'gamma_q' : 0}
    
    initial_matrix = np.zeros((5, 5))
    initial_matrix[0, 0] = 1
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, zeem_par, quad_par,
                                                     initial_state=initial_matrix)
    
    t, signal = FID_Signal(spin, h_unperturbed, dm_0, time_window=100, T2=1)
    
    assert np.absolute(signal[-1])<1e-10
    
# Checks that the Fourier transform of two FID signal acquired with a phase difference of pi are one the
# opposite of the other
def test_Opposite_Decay_Signal():
    spin_par = {'quantum number' : 3,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : math.pi/4,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 5.,
                'asymmetry parameter' : 0.3,
                'alpha_q' : math.pi/3,
                'beta_q' : math.pi/5,
                'gamma_q' : 0}
    
    initial_matrix = np.zeros((7, 7))
    initial_matrix[0, 0] = 1
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, zeem_par, quad_par,
                                                     initial_state=initial_matrix)
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_0, \
                        mode, pulse_time=10, \
                        picture='IP', \
                        n_points=10)
    
    t, signal1 = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=250, T2=100)
    t, signal2 = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=250, T2=100, phi=math.pi)
    
    f, fourier1 = Fourier_Transform_Signal(signal1, t, 7.5, 12.5)
    f, fourier2 = Fourier_Transform_Signal(signal2, t, 7.5, 12.5)
    
    assert np.all(np.isclose(fourier1, -fourier2, rtol=1e-10))

# Checks that the Fourier transform of the signal has the same shape both after adding the phase
# computed by Fourier_Phase_Shift directly to the FID signal and by rotating the detection coil by the
# same angle
def test_Two_Methods_Phase_Adjustment():
    spin_par = {'quantum number' : 3/2,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0,
                'asymmetry parameter' : 0,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    initial_matrix = np.zeros((4, 4))
    initial_matrix[0, 0] = 1
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, zeem_par, quad_par,
                                                     initial_state=initial_matrix)
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_0, \
                        mode, pulse_time=math.pi, \
                        picture='IP', \
                        n_points=10)
    
    t, fid = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=500)
    f, fourier0 = Fourier_Transform_Signal(fid, t, 9, 11)
            
    phi = Fourier_Phase_Shift(f, fourier0, peak_frequency_hint=10)
    f, fourier1 = Fourier_Transform_Signal(np.exp(1j*phi)*fid, t, 9, 11)
            
    t, fid_rephased = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=500, phi=-phi)
    f, fourier2 = Fourier_Transform_Signal(fid_rephased, t, 9, 11)
        
    assert np.all(np.isclose(fourier1, fourier2, rtol=1e-10))


    
    
    
    