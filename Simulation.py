import numpy as np
import pandas as pd
import math

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
                         H_Pulse_IP, \
                         V0, V1, V2

# Function that runs the simulation
def Simulate(spin_par, zeem_par, quad_par, mode, temperature, pulse_time):
    
    # Nuclear spin under study
    spin = Nuclear_Spin(spin_par['quantum number'], \
                        spin_par['gyromagnetic ratio'])
    
    # Zeeman term of the Hamiltonian
    h_zeeman = H_Zeeman(spin, zeem_par['theta_z'], \
                              zeem_par['phi_z'], \
                              zeem_par['field magnitude'])
    
    # Quadrupole term of the Hamiltonian
    h_quadrupole = H_Quadrupole(spin, quad_par['coupling constant'], \
                                      quad_par['asymmetry parameter'], \
                                      quad_par['alpha_q'], \
                                      quad_par['beta_q'], \
                                      quad_par['gamma_q'])
    
    # Computes the unperturbed Hamiltonian of the system, namely the sum of the Zeeman and quadrupole
    # contributions
    h_unperturbed = Observable(h_zeeman.matrix + h_quadrupole.matrix)
    
    Energy_Spectrum(h_unperturbed)
    
    # Density matrix of the system at time t=0, when the ensemble of spins is at equilibrium
    dm_initial = Canonical_Density_Matrix(h_unperturbed, temperature)
        
    # Evolves the density matrix under the action of the specified pulse through the time interval
    # pulse_time
    Evolve(spin, dm_initial, h_unperturbed, mode, pulse_time)

    
def Energy_Spectrum(h_0):
    energy_spectrum = h_0.eigenvalues()
    print("Energy eigenvalues of the Zeeman + quadrupole Hamiltonian = %r" % (energy_spectrum), '\n')
    return energy_spectrum
    

# Computes the density matrix of the system after the application of a desired pulse for a given time,
# given the initial preparation of the ensemble
def Evolve(spin, dm_0, h_0, mode, T):
    
    if T == 0: return dm_0
    
    print("Initial density matrix = \n %r" % (dm_0.matrix), '\n')
    
    # Sampling of the time-dependent term of the Hamiltonian representing the coupling with the
    # electromagnetic pulse (already cast in the interaction picture) in the time window [0, T]
    times, time_step = np.linspace(0, T, num=int(T*100), retstep=True)
    h_pulse_ip = []
    for t in times:
        h_pulse_ip.append(H_Pulse_IP(spin, mode, t, h_0))
        
    # Evaluation of the 1st and 2nd terms of the Magnus expansion for the pulse Hamiltonian in the
    # interaction picture
    magnus_1st = Magnus_Expansion_1st_Term(h_pulse_ip, time_step)
    magnus_2nd = Magnus_Expansion_2nd_Term(h_pulse_ip, time_step)
    
    # Density matrix of the system after evolution under the action of the pulse (in the interaction
    # picture)
    dm_T_ip = dm_0.sim_trans(-(magnus_1st+magnus_2nd), exp=True)
    
    # Evolved density matrix cast back in the Schroedinger picture
    dm_T = dm_T_ip.interaction_picture(h_0, T, invert=True)
        
    print("Evolved density matrix = \n %r" % (dm_T.matrix), '\n')
    
    return Density_Matrix(dm_T.matrix)