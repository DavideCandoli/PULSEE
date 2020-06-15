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

from Simulation import Simulate_Evolution, \
                       Simulate_Transition_Spectrum, \
                       Plot_Real_Density_Matrix, \
                       Plot_Transition_Spectrum


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
    RRF_par = {'omega_RRF': 10,
               'theta_RRF': math.pi,
               'phi_RRF': 0}
    dm_evolved = Simulate_Evolution(spin_par, \
                                    zeem_par, \
                                    quad_par, \
                                    mode=mode, \
                                    temperature=1e-10, \
                                    pulse_time=20, \
                                    picture = 'RRF', \
                                    RRF_par=RRF_par)
    Plot_Real_Density_Matrix(dm_evolved)
    f, p = Simulate_Transition_Spectrum(spin_par, \
                                        zeem_par, \
                                        quad_par, \
                                        mode=mode, \
                                        pulse_time=20)
    Plot_Transition_Spectrum(f, p)
    

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
    RRF_par = {'omega_RRF': 10,
               'theta_RRF': math.pi,
               'phi_RRF': 0}
    dm_evolved = Simulate_Evolution(spin_par, \
                                    zeem_par, \
                                    quad_par, \
                                    mode=mode, \
                                    temperature=1e-10, \
                                    pulse_time=20, \
                                    picture = 'RRF', \
                                    RRF_par=RRF_par)
    Plot_Real_Density_Matrix(dm_evolved)
    f, p = Simulate_Transition_Spectrum(spin_par, \
                                        zeem_par, \
                                        quad_par, \
                                        mode=mode, \
                                        pulse_time=20)
    Plot_Transition_Spectrum(f, p)


def Spectrum_Pure_Symmetric_Quadrupole():
    spin_par = {'quantum number' : 2,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 0.,
                'theta_z' : 0,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 8,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    mode = pd.DataFrame([(1., 1., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    RRF_par = {'omega_RRF': 1.,
               'theta_RRF': 0,
               'phi_RRF': 0}
    dm_evolved = Simulate_Evolution(spin_par, \
                                    zeem_par, \
                                    quad_par, \
                                    mode=mode, \
                                    temperature=1e-11, \
                                    pulse_time=20, \
                                    picture = 'RRF', \
                                    RRF_par=RRF_par)
    Plot_Real_Density_Matrix(dm_evolved)
    f, p = Simulate_Transition_Spectrum(spin_par, \
                                        zeem_par, \
                                        quad_par, \
                                        mode=mode, \
                                        pulse_time=20)
    Plot_Transition_Spectrum(f, p)


def Spectrum_Pure_Asymmetric_Quadrupole_Integer_Spin():
    spin_par = {'quantum number' : 1,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 0.,
                'theta_z' : 0,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 10,
                'asymmetry parameter' : 0.6,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    mode = pd.DataFrame([(10, 1., 0., math.pi/4, math.pi/4)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    RRF_par = {'omega_RRF': 10,
               'theta_RRF': -math.pi/4,
               'phi_RRF': 0}
    dm_evolved = Simulate_Evolution(spin_par, \
                                    zeem_par, \
                                    quad_par, \
                                    mode=mode, \
                                    temperature=1e-10, \
                                    pulse_time=20, \
                                    picture = 'RRF', \
                                    RRF_par=RRF_par)
    Plot_Real_Density_Matrix(dm_evolved)
    f, p = Simulate_Transition_Spectrum(spin_par, \
                                        zeem_par, \
                                        quad_par, \
                                        mode=mode, \
                                        pulse_time=20)
    Plot_Transition_Spectrum(f, p)
    
    
def Spectrum_Pure_Asymmetric_Quadrupole_Half_Integer_Spin():
    spin_par = {'quantum number' : 3/2,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 0.,
                'theta_z' : 0,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 10*math.sqrt(3),
                'asymmetry parameter' : 1,
                'alpha_q' : 0.,
                'beta_q' : 0.,
                'gamma_q' : 0.}
    mode = pd.DataFrame([(10, 3., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    RRF_par = {'omega_RRF': 10,
               'theta_RRF': math.pi,
               'phi_RRF': 0}
    dm_evolved = Simulate_Evolution(spin_par, \
                                    zeem_par, \
                                    quad_par, \
                                    mode=mode, \
                                    temperature=1e-10, \
                                    pulse_time=20, \
                                    picture = 'RRF', \
                                    RRF_par=RRF_par)
    Plot_Real_Density_Matrix(dm_evolved)
    f, p = Simulate_Transition_Spectrum(spin_par, \
                                        zeem_par, \
                                        quad_par, \
                                        mode=mode, \
                                        pulse_time=20)
    Plot_Transition_Spectrum(f, p)

    
def Spectrum_Perturbed_Quadrupole_Integer_Spin():
    spin_par = {'quantum number' : 1,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : math.sqrt(2),
                'theta_z' : math.pi/4,
                'phi_z' : 0}
    quad_par = {'coupling constant' : 8,
                'asymmetry parameter' : 0,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    mode = pd.DataFrame([(10, 1., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    RRF_par = {'omega_RRF': 10,
               'theta_RRF': 0,
               'phi_RRF': 0}
    dm_evolved = Simulate_Evolution(spin_par, \
                                    zeem_par, \
                                    quad_par, \
                                    mode=mode, \
                                    temperature=1e-10, \
                                    pulse_time=20, \
                                    picture = 'RRF', \
                                    RRF_par=RRF_par)
    Plot_Real_Density_Matrix(dm_evolved)
    f, p = Simulate_Transition_Spectrum(spin_par, \
                                        zeem_par, \
                                        quad_par, \
                                        mode=mode, \
                                        pulse_time=20)
    Plot_Transition_Spectrum(f, p)

    
def Spectrum_Perturbed_Quadrupole_Half_Integer_Spin():
    spin_par = {'quantum number' : 3/2,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 1/(math.cos(math.atan(math.sqrt(3)/2))),
                'theta_z' : math.atan(math.sqrt(3)/2),
                'phi_z' : 0}
    quad_par = {'coupling constant' : 12,
                'asymmetry parameter' : 0,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    mode = pd.DataFrame([(10, 1., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    RRF_par = {'omega_RRF': 10,
               'theta_RRF': 0,
               'phi_RRF': 0}
    dm_evolved = Simulate_Evolution(spin_par, \
                                    zeem_par, \
                                    quad_par, \
                                    mode=mode, \
                                    temperature=1e-12, \
                                    pulse_time=20, \
                                    picture = 'RRF', \
                                    RRF_par=RRF_par)
    Plot_Real_Density_Matrix(dm_evolved)
    f, p = Simulate_Transition_Spectrum(spin_par, \
                                        zeem_par, \
                                        quad_par, \
                                        mode=mode, \
                                        pulse_time=20)
    Plot_Transition_Spectrum(f, p)

    