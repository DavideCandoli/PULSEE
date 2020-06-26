import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import xticks, yticks

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

# Sets the following elements of the system under study:
# - Nuclear spin
# - Unperturbed Hamiltonian
# - Initial density matrix (canonical)
# and returns the density matrix of the system after a time pulse_time.
def Simulate_Evolution(spin_par, zeem_par, quad_par, mode, \
                       temperature, pulse_time, \
                       picture='RRF', RRF_par={'omega_RRF': 0,
                                               'theta_RRF': 0,
                                               'phi_RRF': 0}, \
                       initial_state = 'canonical', \
                       n_sample_points=10):
    
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
    
    # Density matrix of the system at time t=0, when the ensemble of spins is at equilibrium
    if initial_state == 'canonical': dm_initial = Canonical_Density_Matrix(h_unperturbed, temperature)
    else: dm_initial = initial_state
    
    # Selects the operator for the change of picture, according to the value of parameter 'picture'
    if picture == 'IP':
        o_change_of_picture = h_unperturbed
    else:
        o_change_of_picture = RRF_Operator(spin, RRF_par)
    
    # Evolves the density matrix under the action of the specified pulse through the time interval
    # pulse_time, performing the evolution in the specified picture
    dm_evolved = Evolve(spin, dm_initial, h_unperturbed, mode, pulse_time, o_change_of_picture, n_sample_points)
            
    return dm_evolved


# Sets the following elements of the system under study:
# - Nuclear spin
# - Unperturbed Hamiltonian
# and returns the spectrum of the transitions induced between the energy eigenstates of the system
# through a time pulse_time
def Simulate_Transition_Spectrum(spin_par, zeem_par, quad_par, mode, pulse_time):
    
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
    
    # Computes the frequencies and probabilities of transition induced by the pulse under consideration
    # between the eigenstates of h_unperturbed
    t_frequencies, t_probabilities = Transition_Spectrum(spin, h_unperturbed, mode, pulse_time)
    
    return t_frequencies, t_probabilities


# Computes the spectrum of the transitions induced by the pulse specified by 'mode' between the
# eigenstates of h_unperturbed after a time T
def Transition_Spectrum(spin, h_unperturbed, mode, T):
    
    # Energy levels and eigenstates of the unperturbed Hamiltonian
    energies, o_change_of_basis = h_unperturbed.diagonalise()
    
    # Hamiltonian of the pulse evaluated at time T
    h_pulse_T = H_Multiple_Mode_Pulse(spin, mode, T)
    
    transition_frequency = []
    
    transition_probability = []
    
    d = h_unperturbed.dimension()
    
    # In the following loop, the frequencies and the respective probabilities of transition are computed
    # and recorded in the appropriate lists
    for i in range(d):
        for j in range(d):
            if i < j:
                transition_frequency.append(np.absolute(energies[j] - energies[i]))
                h_pulse_T_eig = h_pulse_T.sim_trans(o_change_of_basis)
                transition_probability.append((np.absolute(h_pulse_T_eig.matrix[j][i]))**2)
            else:
                pass
    
    return transition_frequency, transition_probability


def Plot_Transition_Spectrum(frequencies, probabilities, save=False, name='TransitionSpectrum', destination=''):
    plt.vlines(frequencies, 0, probabilities, colors='b')
    
    plt.xlabel("\N{GREEK SMALL LETTER OMEGA} (MHz)")    
    plt.ylabel("Probability (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    plt.show()


# Operator which generates a change of picture equivalent to moving to the rotating reference frame
# (RRF)
def RRF_Operator(spin, RRF_par):
    omega = RRF_par['omega_RRF']
    theta = RRF_par['theta_RRF']
    phi = RRF_par['phi_RRF']
    RRF_operator = omega*(spin.I['z']*math.cos(theta) + \
                          spin.I['x']*math.sin(theta)*math.cos(phi) + \
                          spin.I['y']*math.sin(theta)*math.sin(phi))
    return Observable(RRF_operator.matrix)


# Computes the density matrix of the system after the application of a desired pulse for a given time, 
# given the initial preparation of the ensemble. The evolution is performed in the picture generated by
# o_change_of_picture
def Evolve(spin, dm_0, h_0, mode, T, o_change_of_picture, n_points=10):
    
    if T == 0:
        return dm_0
    
    # Sampling of the Hamiltonian in the desired picture over the time window [0, T]
    times, time_step = np.linspace(0, T, num=int(T*n_points), retstep=True)
    h_ip = []
    for t in times:
        h_ip.append(H_Changed_Picture(spin, mode, h_0, o_change_of_picture, t))
    
    # Evaluation of the 1st and 2nd terms of the Magnus expansion for the Hamiltonian in the new picture
    magnus_1st = Magnus_Expansion_1st_Term(h_ip, time_step)
    magnus_2nd = Magnus_Expansion_2nd_Term(h_ip, time_step)

    # Density matrix of the system after evolution under the action of the pulse, expressed
    # in the new picture
    dm_T_ip = dm_0.sim_trans(-(magnus_1st+magnus_2nd), exp=True)

    # Evolved density matrix cast back in the Schroedinger picture
    dm_T = dm_T_ip.change_picture(o_change_of_picture, T, invert=True)
    
    return Density_Matrix(dm_T.matrix)


# Returns the free induction decay (FID) signal resulting from the free evolution of the component
# of the magnetization on the x-y plane of the LAB system. The initial state of the system is given by
# the parameter dm, and the dynamics of the magnetization is recorded for a time final_time. Relaxation
# effects are not taken into account.
def FID_Signal(spin_par, dm, zeem_par, quad_par, final_time):
    
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
    
    # Sampling of the time window [0, final_time] (microseconds) where the free evolution takes place
    times = np.linspace(start=0, stop=final_time, num=final_time*10)
    
    # FID signal to be sampled
    FID = []
    
    # Computes the FID assuming that the detection coil records the time-dependence of the magnetization
    # on the x-y plane, given by
    # S = Tr[dm(t)*I-]
    for t in times:
        dm_t = dm.free_evolution(h_unperturbed, t)
        FID.append((dm_t*spin.I['-']).trace())
    
    return times, FID


# Generates a 3D histogram of the real part of the passed density matrix
def Plot_Real_Density_Matrix(dm, save=False, name='RealPartDensityMatrix', destination=''):
    
    # Retain only the real part of the density matrix elements
    real_part = np.vectorize(np.real)
    data_array = real_part(dm.matrix)
    
    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data. You can think of this as the floor of the
    # plot.
    x_data, y_data = np.meshgrid(np.arange(data_array.shape[1])+0.25,
                                 np.arange(data_array.shape[0])+0.25)
    
    
    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays:
    # x_data, y_data, z_data. The following call boils down to picking
    # one entry from each array and plotting a bar to from
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
    ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             dx, dy, z_data)
    
    # Labels of the plot
    
    s = (data_array.shape[0]-1)/2

    xticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), np.arange(start=s, stop=-s-1, step=-1))
    yticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), np.arange(start=s, stop=-s-1, step=-1))
    
    ax.set_xlabel("m")    
    ax.set_ylabel("m")
    ax.set_zlabel("Re(\N{GREEK SMALL LETTER RHO})")
    
    # Save the plot
    
    if save: plt.savefig(destination + name)

    # Finally, display the plot.
    
    plt.show()