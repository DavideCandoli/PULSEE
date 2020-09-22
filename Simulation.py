import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import xticks, yticks

from Operators import *

from Nuclear_Spin import *

from Hamiltonians import *


def nuclear_system_setup(spin_par, zeem_par, quad_par, initial_state='canonical', temperature=1e-4):
    
    spin = Nuclear_Spin(spin_par['quantum number'], \
                        spin_par['gamma/2pi'])
    
    h_z = h_zeeman(spin, zeem_par['theta_z'], \
                         zeem_par['phi_z'], \
                         zeem_par['field magnitude'])
    
    h_q = h_quadrupole(spin, quad_par['coupling constant'], \
                             quad_par['asymmetry parameter'], \
                             quad_par['alpha_q'], \
                             quad_par['beta_q'], \
                             quad_par['gamma_q'])
    
    h_unperturbed = Observable(h_z.matrix + h_q.matrix)
    
    if isinstance(initial_state, str) and initial_state == 'canonical':
        dm_initial = canonical_density_matrix(h_unperturbed, temperature)
    else:
        dm_initial = Density_Matrix(initial_state)
    
    return spin, h_unperturbed, dm_initial


def evolve(spin, h_unperturbed, dm_initial, \
           mode, pulse_time, \
           picture='RRF', RRF_par={'nu_RRF': 0,
                                   'theta_RRF': 0,
                                   'phi_RRF': 0}, \
           n_points=10, order=2):
    
    if pulse_time == 0 or np.all(np.absolute((dm_initial-Operator(spin.d)).matrix)<1e-10):
        return dm_initial
    
    if picture == 'IP':
        o_change_of_picture = h_unperturbed
    else:
        o_change_of_picture = RRF_Operator(spin, RRF_par)
    
    times, time_step = np.linspace(0, pulse_time, num=int(pulse_time*n_points), retstep=True)
    h_new_picture = []
    for t in times:
        h_new_picture.append(h_changed_picture(spin, mode, h_unperturbed, o_change_of_picture, t))
    
    magnus_exp = magnus_expansion_1st_term(h_new_picture, time_step)
    if order>1:
        magnus_exp = magnus_exp + magnus_expansion_2nd_term(h_new_picture, time_step)
        if order>2:
            magnus_exp = magnus_exp + magnus_expansion_3rd_term(h_new_picture, time_step)

    dm_evolved_new_picture = dm_initial.sim_trans(-magnus_exp, exp=True)

    dm_evolved = dm_evolved_new_picture.changed_picture(o_change_of_picture, pulse_time, invert=True)
    
    return Density_Matrix(dm_evolved.matrix)


# Operator which generates a change of picture equivalent to moving to the rotating reference frame
# (RRF)
def RRF_Operator(spin, RRF_par):
    nu = RRF_par['nu_RRF']
    theta = RRF_par['theta_RRF']
    phi = RRF_par['phi_RRF']
    RRF_operator = nu*(spin.I['z']*math.cos(theta) + \
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
    energies, o_change_of_basis = h_unperturbed.diagonalisation()
    
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
                magnetization_eig = spin.gyro_ratio_over_2pi*spin.I['x'].sim_trans(o_change_of_basis)
                
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
    # F = int_0^T{exp(i 2pi nu t) S(t) dt}
    for nu in frequencies:
        integral = 0
        for i in range(len(times)):
            integral = integral + np.exp(1j*2*math.pi*nu*times[i])*signal[i]*dt
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
    if im >= 0:
        phase = math.atan(-im/re)
    else:
        phase = math.atan(-im/re) + math.pi
    
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








