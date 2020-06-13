import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term, \
                      Canonical_Density_Matrix

from Nuclear_Spin import Nuclear_Spin

from Hamiltonians import H_Zeeman, H_Quadrupole, \
                         H_Single_Mode_Pulse, \
                         H_Multiple_Mode_Pulse, \
                         H_Changed_Picture, \
                         V0, V1, V2

from Simulation import Evolve, Transition_Spectrum, Simulate


def Spectrum_Pure_Zeeman():
    spin_par = {'quantum number' : 3/2,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    Simulate(spin_par, \
             zeem_par, \
             quad_par, \
             mode=mode, \
             temperature=300, \
             pulse_time=20)
    

def Spectrum_Perturbed_Zeeman():
    spin_par = {'quantum number' : 5/2,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 1.,
                'asymmetry parameter' : 0.,
                'alpha_q' : math.pi/4,
                'beta_q' : math.pi/4,
                'gamma_q' : math.pi/4}
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    Simulate(spin_par, \
             zeem_par, \
             quad_par, \
             mode=mode, \
             temperature=300, \
             pulse_time=20)


def Spectrum_Pure_Symmetric_Quadrupole():
    spin_par = {'quantum number' : 2,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 0.,
                'theta_z' : 0,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 8,
                'asymmetry parameter' : 0.,
                'alpha_q' : math.pi/4,
                'beta_q' : math.pi/4,
                'gamma_q' : math.pi/4}
    mode = pd.DataFrame([(10, 1., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    Simulate(spin_par, \
             zeem_par, \
             quad_par, \
             mode=mode, \
             temperature=300, \
             pulse_time=20)


def Spectrum_Pure_Asymmetric_Quadrupole_Integer_Spin():
    spin_par = {'quantum number' : 1,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 0.,
                'theta_z' : 0,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 10,
                'asymmetry parameter' : 0.6,
                'alpha_q' : math.pi/4,
                'beta_q' : math.pi/4,
                'gamma_q' : math.pi/4}
    mode = pd.DataFrame([(10, 1., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    Simulate(spin_par, \
             zeem_par, \
             quad_par, \
             mode=mode, \
             temperature=300, \
             pulse_time=20)

