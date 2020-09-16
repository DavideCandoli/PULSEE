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


# Sets up and returns the following elements of the system under study:
# - Nuclear spin
# - Unperturbed Hamiltonian
# - State of the system at t=0
def Nuclear_System_Setup(spin_par, zeem_par, quad_par, initial_state='canonical', temperature=1e-4):
    
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
    
    # Sets the density matrix of the system at time t=0, according to the value of 'initial_state'
    if isinstance(initial_state, str) and initial_state == 'canonical':
        dm_initial = Canonical_Density_Matrix(h_unperturbed, temperature)
    else:
        dm_initial = Density_Matrix(initial_state)
    
    return spin, h_unperturbed, dm_initial


# Computes the density matrix of the system after the application of a desired pulse for a given time, 
# given the initial preparation of the ensemble. The evolution is performed in the picture specified by
# the argument
def Evolve(spin, h_unperturbed, dm_initial, \
           mode, pulse_time, \
           picture='RRF', RRF_par={'omega_RRF': 0,
                                   'theta_RRF': 0,
                                   'phi_RRF': 0}, \
           n_points=10):
    
    # Selects the operator for the change of picture, according to the value of 'picture'
    if picture == 'IP':
        o_change_of_picture = h_unperturbed
    else:
        o_change_of_picture = RRF_Operator(spin, RRF_par)
    
    # Returns the same density matrix as the initial one when the passed pulse time is exactly 0
    if pulse_time == 0:
        return dm_initial
    
    # Sampling of the Hamiltonian in the desired picture over the time window [0, pulse_time]
    times, time_step = np.linspace(0, pulse_time, num=int(pulse_time*n_points), retstep=True)
    h_ip = []
    for t in times:
        h_ip.append(H_Changed_Picture(spin, mode, h_unperturbed, o_change_of_picture, t))
    
    # Evaluation of the 1st and 2nd terms of the Magnus expansion for the Hamiltonian in the new picture
    magnus_1st = Magnus_Expansion_1st_Term(h_ip, time_step)
    magnus_2nd = Magnus_Expansion_2nd_Term(h_ip, time_step)

    # Density matrix of the system after evolution under the action of the pulse, expressed
    # in the new picture
    dm_evolved_ip = dm_initial.sim_trans(-(magnus_1st+magnus_2nd), exp=True)

    # Evolved density matrix cast back in the Schroedinger picture
    dm_evolved = dm_evolved_ip.change_picture(o_change_of_picture, pulse_time, invert=True)
    
    return Density_Matrix(dm_evolved.matrix)


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


# Generates a 3D histogram of the real part of the passed density matrix
def Plot_Real_Density_Matrix(dm, show=True, save=False, name='RealPartDensityMatrix', destination=''):
    
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
    # one entry from each array and plotting a bar from
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
    
    if save:
        plt.savefig(destination + name)

    # Finally, display the plot.
    
    if show:
        plt.show()
        
    return fig
    

# Computes the spectrum of power absorption due to x-polarized single-mode pulses, appealing to the
# formula derived using Fermi's golden rule. If normalized=False, the state of initial preparation of
# the ensemble is taken into account for the calculation of the spectrum
def Transition_Spectrum(spin, h_unperturbed, normalized=True, dm_initial=0):
    
    # Energy levels and eigenstates of the unperturbed Hamiltonian
    energies, o_change_of_basis = h_unperturbed.diagonalise()
    
    transition_frequency = []
    
    transition_probability = []
    
    d = h_unperturbed.dimension()
    
    # In the following loop, the frequencies and the respective intensities of the spectrum are computed
    # and recorded in the appropriate lists
    for i in range(d):
        for j in range(d):
            if i < j:
                omega = np.absolute(energies[j] - energies[i])
                transition_frequency.append(omega)
                magnetization_eig = spin.gyromagnetic_ratio*spin.I['x'].sim_trans(o_change_of_basis)
                
                P_omega = omega*(np.absolute(magnetization_eig.matrix[j, i]))**2
                
                if not normalized:
                    p_i = dm_initial.matrix[i, i]
                    p_j = dm_initial.matrix[j, j]
                    P_omega = np.absolute(p_i-p_j)*P_omega
                    
                transition_probability.append(P_omega)
            else:
                pass
    
    return transition_frequency, transition_probability


def Plot_Transition_Spectrum(frequencies, probabilities, show=True, save=False, name='TransitionSpectrum', destination=''):
    fig = plt.figure()
    
    plt.vlines(frequencies, 0, probabilities, colors='b')
    
    plt.xlabel("\N{GREEK SMALL LETTER OMEGA} (MHz)")    
    plt.ylabel("Probability (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    if show: plt.show()
        
    return fig


# Returns the free induction decay (FID) signal resulting from the free evolution of the component
# of the magnetization on the plane specified by theta, phi of the LAB system. The state of the system
# at the beginning of acquisition is given by the parameter dm, and the dynamics of the magnetization is 
# recorded for a time time_window. The relaxation time T2 is passed as an argument.
def FID_Signal(spin, h_unperturbed, dm, time_window, T2=100, theta=0, phi=0):
    
    # Sampling of the time window [0, time_window] (microseconds) where the free evolution takes place
    times = np.linspace(start=0, stop=time_window, num=int(time_window*10))
    
    # FID signal to be sampled
    FID = []
    
    # Computes the FID assuming that the detection coil records the time-dependence of the magnetization
    # on the plane perpendicular to (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)), given by
    # FID = Tr[dm(t) e^{i phi I_z} e^{i theta I_y} I+ e^{-i theta I_y} e^{-i phi I_z} e^{-t/T2}]
    for t in times:
        dm_t = dm.free_evolution(h_unperturbed, t)
        Iz = spin.I['z']
        Iy = spin.I['y']
        I_plus_rotated = (1j*phi*Iz).exp()*(1j*theta*Iy).exp()*spin.I['+']*(-1j*theta*Iy).exp()*(-1j*phi*Iz).exp()
        FID.append((dm_t*I_plus_rotated).trace()*np.exp(-t/T2))
    
    return times, np.array(FID)


# Plots the the real part of the FID signal as a function of time
def Plot_FID_Signal(times, FID, show=True, save=False, name='FIDSignal', destination=''):
    fig = plt.figure()
    
    plt.plot(times, np.real(FID))
    
    plt.xlabel("time (\N{GREEK SMALL LETTER MU}s)")    
    plt.ylabel("FID (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    if show: plt.show()
        
    return fig


# Computes the complex Fourier transform of the given signal originally expressed in the time domain
def Fourier_Transform_Signal(signal, times, frequency_start, frequency_stop):
    
    # Whole duration of the signal
    T = times[-1]
    
    # Step between the sampled instants of time
    dt = times[1]-times[0]
    
    # Values of frequency at which the Fourier transform is to be evaluated (one-signed)
    frequencies = np.linspace(start=frequency_start, stop=frequency_stop, num=1000)
    
    # Fourier transform to be sampled
    fourier = []
    
    # The Fourier transform is evaluated through the conventional formula
    # F = int_0^T{exp(-i omega t) S(t) dt}
    for omega in frequencies:
        integral = 0
        for i in range(len(times)):
            integral = integral + np.exp(-1j*omega*times[i])*signal[i]*dt
        fourier.append(integral)
    
    return frequencies, np.array(fourier)


# Finds out the phase responsible for the displacement of the real and imaginary parts of the Fourier
# spectrum of the FID with respect to the ideal absorptive/dispersive shapes
def Fourier_Phase_Shift(frequencies, fourier, peak_frequency_hint, search_window=0.1):
    # Position of the specified peak in the list frequencies
    peak_pos = 0
    
    # Range where to look for the maximum of the square modulus of the Fourier spectrum
    search_range = np.nonzero(np.isclose(frequencies, peak_frequency_hint, atol=search_window/2))[0]
    
    # Search of the maximum of the square modulus of the Fourier spectrum
    fourier2_max=0
    for i in search_range:
        if (np.absolute(fourier[i])**2)>fourier2_max:
            fourier2_max = np.absolute(fourier[i])
            peak_pos = i
        
    # Real part of the Fourier spectrum at the peak
    re = np.real(fourier[peak_pos])
    
    # Imaginary part of the Fourier spectrum at the peak
    im = np.imag(fourier[peak_pos])
    
    # Phase shift
    phase = math.atan(-im/re)
    
    return phase


# Plots the Fourier transform of the signal
def Plot_Fourier_Transform(frequencies, fourier, square_modulus=False, show=True, save=False, name='FTSignal', destination=''):
    fig = plt.figure()
    
    if not square_modulus:
        plt.plot(frequencies, np.real(fourier), label='Real part')
    
        plt.plot(frequencies, np.imag(fourier), label='Imaginary part')
    
    else:
        plt.plot(frequencies, np.absolute(fourier)**2, label='Square modulus')
    
    plt.legend(loc='upper left')
    
    plt.xlabel("frequency (MHz)")    
    plt.ylabel("FT signal (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    if show: plt.show()
        
    return fig








