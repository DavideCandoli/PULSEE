import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt

from Operators import *

from Nuclear_Spin import *

from Hamiltonians import *

from Simulation import *


# Plots:
# - The density matrix evolved under a rf pulse
# - The spectrum of the transitions induced by the rf pulse according to Fermi golden rule
# - The Fourier spectrum of the FID signal
# for a pure Zeeman Hamiltonian nucleus
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
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0),
                         (10., 1., math.pi/2, math.pi/2, -math.pi/2)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    RRF_par = {'nu_RRF': 10,
               'theta_RRF': math.pi,
               'phi_RRF': 0}
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par)
    
    Plot_Real_Density_Matrix(dm_0, save=False, name='DMPureZeeman')
    
    dm_evolved = Evolve(spin, h_unperturbed, dm_0, \
                        mode=mode, pulse_time=1/4, \
                        picture = 'IP', RRF_par=RRF_par)
    
    Plot_Real_Density_Matrix(dm_evolved, save=False, name='DMPureZeeman')
    
    f, p = Transition_Spectrum(spin, h_unperturbed, normalized=True)
    
    Plot_Transition_Spectrum(f, p, save=False, name='SpectrumPureZeeman')
    
    t, FID = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=500, phi=math.pi/2)
        
    f, ft = Fourier_Transform_Signal(FID, t, 9.5, 10.5)
    
    Plot_Fourier_Transform(f, ft)
    
    Plot_Fourier_Transform(f, ft, square_modulus=True)


# Plots:
# - The density matrix evolved under a rf pulse
# - The spectrum of the transitions induced by the rf pulse according to Fermi golden rule
# - The Fourier spectrum of the FID signal
# for a nucleus where the quadrupole interaction is a perturbation of the Zeeman energy
def Spectrum_Perturbed_Zeeman():
    spin_par = {'quantum number' : 5/2,
                'gyromagnetic ratio' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 2.,
                'asymmetry parameter' : 0.,
                'alpha_q' : math.pi/4,
                'beta_q' : math.pi/4,
                'gamma_q' : math.pi/4}
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)],
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    RRF_par = {'nu_RRF': 0,
               'theta_RRF': 0,
               'phi_RRF': 0}
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par, \
                                                     initial_state='canonical', temperature=300)
    
    dm_evolved = Evolve(spin, h_unperturbed, dm_0, \
                        mode=mode, pulse_time=0.5, \
                        picture = 'IP', RRF_par=RRF_par)
    
    Plot_Real_Density_Matrix(dm_evolved, save=False, name='DMPerturbedZeeman')
    
    f, p = Transition_Spectrum(spin, h_unperturbed)
    
    Plot_Transition_Spectrum(f, p, save=False, name='SpectrumPerturbedZeeman')
    
    t, FID = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=2000, T2=500, phi=math.pi/2)
    
    f, ft = Fourier_Transform_Signal(FID, t, 9.8, 10.2)
    
    Plot_Fourier_Transform(f, ft)
    
    Plot_Fourier_Transform(f, ft, square_modulus=True)


# Plots the transition spectrum for a pure quadrupole Hamiltonian, where the EFG is axially symmetric
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
    
    mode = pd.DataFrame([(3., 1., 0., math.pi/2, 0),
                         (3., 1., math.pi/2., math.pi/2, math.pi/2)],
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    RRF_par = {'nu_RRF': 3.,
               'theta_RRF': math.pi,
               'phi_RRF': 0}
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par, \
                                                     initial_state='canonical', temperature=300)
    
    Plot_Real_Density_Matrix(dm_0, save=False, name='DMPureSymmetricQuadrupole')
    
    dm_evolved = Evolve(spin, h_unperturbed, dm_0, \
                        mode=mode, pulse_time=2*math.pi/3, \
                        picture = 'IP', RRF_par=RRF_par, \
                        n_points=10)
    
    Plot_Real_Density_Matrix(dm_evolved, save=False, name='DMPureSymmetricQuadrupole')
    
    f, p = Transition_Spectrum(spin, h_unperturbed)
    
    Plot_Transition_Spectrum(f, p, save=False, name='SpectrumPureSymmetricQuadrupole')
    
    t, FID = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=500)
        
    f, ft = Fourier_Transform_Signal(FID, t, 0, 4)
    
    Plot_Fourier_Transform(f, ft)
    
    Plot_Fourier_Transform(f, ft, square_modulus=True)


# Plots the transition spectrum of an integer spin nucleus with a pure quadrupole Hamiltonian where the
# EFG is axially asymmetric
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
    
    mode = pd.DataFrame([(10, 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    RRF_par = {'nu_RRF': 10,
               'theta_RRF': math.pi,
               'phi_RRF': 0}
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par, \
                                                     initial_state='canonical', temperature=1e-3)
    
    dm_evolved = Evolve(spin, h_unperturbed, dm_0, \
                        mode=mode, pulse_time=20, \
                        picture = 'RRF', RRF_par=RRF_par)
    
    Plot_Real_Density_Matrix(dm_evolved, save=False, name='DMPureAsymmetricQuadrupoleInt')
    
    f, p = Transition_Spectrum(spin, h_unperturbed)
    
    Plot_Transition_Spectrum(f, p, save=False, name='SpectrumPureAsymmetricQuadrupoleInt')
    
    t, FID = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=500)
    
    f, ft = Fourier_Transform_Signal(FID, t, 0, 10)
    
    Plot_Fourier_Transform(f, ft)
    
    Plot_Fourier_Transform(f, ft, square_modulus=True)


# Plots the transition spectrum of a half-integer spin nucleus with a pure quadrupole Hamiltonian where
# the EFG is axially asymmetric
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
    
    mode = pd.DataFrame([(10, 3, 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    RRF_par = {'nu_RRF': 10,
               'theta_RRF': math.pi,
               'phi_RRF': 0}
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par, \
                                                     initial_state='canonical', temperature=1e-6)
    
    dm_evolved = Evolve(spin, h_unperturbed, dm_0, \
                        mode=mode, pulse_time=20, \
                        picture = 'RRF', RRF_par=RRF_par)
    
    Plot_Real_Density_Matrix(dm_evolved, save=False, name='DMPureAsymmetricQuadrupoleHalfInt')
    
    f, p = Transition_Spectrum(spin, h_unperturbed)
    
    Plot_Transition_Spectrum(f, p, save=False, name='SpectrumPureAsymmetricQuadrupoleHalfInt')
    
    t, FID = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=500)
    
    f, ft = Fourier_Transform_Signal(FID, t, 9, 11)
    
    Plot_Fourier_Transform(f, ft)
    
    Plot_Fourier_Transform(f, ft, square_modulus=True)


# Plots the transition spectrum of an integer spin nucleus with a Zeeman perturbed quadrupole
# Hamiltonian where the EFG is axially symmetric
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
    
    RRF_par = {'nu_RRF': 10,
               'theta_RRF': 0,
               'phi_RRF': 0}
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par, \
                                                     initial_state='canonical', temperature=1e-4)
    
    dm_evolved = Evolve(spin, h_unperturbed, dm_0, \
                        mode=mode, pulse_time=20, \
                        picture = 'RRF', RRF_par=RRF_par)
    
    Plot_Real_Density_Matrix(dm_evolved, save=False, name='DMSpectrumPerturbedQuadrupoleInt')
    
    f, p = Transition_Spectrum(spin, h_unperturbed)
    
    Plot_Transition_Spectrum(f, p, save=False, name='SpectrumPerturbedQuadrupoleInt')
    
    t, FID = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=500)
        
    f, ft = Fourier_Transform_Signal(FID, t, 1, 8)
    
    Plot_Fourier_Transform(f, ft)
    
    Plot_Fourier_Transform(f, ft, square_modulus=True)


# Plots the transition spectrum of a half-integer spin nucleus with a Zeeman perturbed quadrupole
# Hamiltonian where the EFG is axially symmetric
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
    
    RRF_par = {'nu_RRF': 10,
               'theta_RRF': 0,
               'phi_RRF': 0}
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par, \
                                                     initial_state='canonical', temperature=1e-3)    
    dm_evolved = Evolve(spin, h_unperturbed, dm_0, \
                        mode=mode, pulse_time=20, \
                        picture = 'RRF', RRF_par=RRF_par)
    
    Plot_Real_Density_Matrix(dm_evolved, save=False, name='DMSpectrumPerturbedQuadrupoleHalfInt')
    
    f, p = Transition_Spectrum(spin, h_unperturbed)
    
    Plot_Transition_Spectrum(f, p, save=False, name='SpectrumPerturbedQuadrupoleHalfInt')
    
    t, FID = FID_Signal(spin, h_unperturbed, dm_evolved, time_window=1000)
    
    f, ft = Fourier_Transform_Signal(FID, t, 1, 10)
    
    Plot_Fourier_Transform(f, ft)
    
    Plot_Fourier_Transform(f, ft, square_modulus=True)
    
