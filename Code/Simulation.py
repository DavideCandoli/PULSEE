import numpy as np
import pandas as pd
import math
from fractions import Fraction

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import xticks, yticks
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from Operators import Operator, Density_Matrix, Observable, \
                      magnus_expansion_1st_term, \
                      magnus_expansion_2nd_term, \
                      magnus_expansion_3rd_term, \
                      canonical_density_matrix

from Many_Body import tensor_product

from Nuclear_Spin import Nuclear_Spin, Many_Spins

from Hamiltonians import h_zeeman, h_quadrupole, \
                         v0_EFG, v1_EFG, v2_EFG, \
                         h_single_mode_pulse, \
                         h_multiple_mode_pulse, \
                         h_changed_picture, \
                         h_j_coupling


def nuclear_system_setup(spin_par, quad_par=None, zeem_par=None, j_matrix=None, initial_state='canonical', temperature=1e-4):
    """
    Sets up the nuclear system under study, returning the objects representing the spin (either a single one or a multiple spins' system), the unperturbed Hamiltonian (made up of the Zeeman, quadrupolar and J-coupling contributions) and the initial state of the system.

    Parameters
    ----------
    - spin_par: dict / list of dict
  
      Map/list of maps containing information about the nuclear spin/spins under consideration. The keys and values required to each dictionary in this argument are shown in the table below.
  
      |           key          |         value        |
      |           ---          |         -----        |
      |    'quantum number'    |  half-integer float  |
      |       'gamma/2pi'      |         float        |
    
      The second item is the gyromagnetic ratio over 2 pi, measured in MHz/T.

    - quad_par: dict / list of dict
    
      Map/maps containing information about the quadrupolar interaction between the electric quadrupole moment and the EFG for each nucleus in the system. The keys and values required to each dictionary in this argument are shown in the table below:
  
      |           key           |       value        |
      |           ---           |       -----        |
      |   'coupling constant'   |       float        |
      |  'asymmetry parameter'  |   float in [0, 1]  |
      |        'alpha_q'        |       float        |
      |        'beta_q'         |       float        |
      |        'gamma_q'        |       float        |
    
    where 'coupling constant' stands for the product e2qQ in the expression of the quadrupole term of the Hamiltonian (to be provided in MHz), 'asymmetry parameter' refers to the same-named property of the EFG, and 'alpha_q', 'beta_q' and 'gamma_q' are the Euler angles for the conversion from the LAB coordinate system to the system of the principal axes of the EFG tensor (PAS) (to be expressed in radians).
    
      When it is None, the quadrupolar interaction of all the spins in the system is not taken into account.
      Default value is None.
    
    - zeem_par: dict
       
      Map containing information about the magnetic field interacting with the magnetic moment of each nucleus in the system. The keys and values required to this argument are shown in the table below:

      |         key         |       value      |
      |         ---         |       -----      |
      |      'theta_z'      |       float      |
      |       'phi_z'       |       float      |
      |  'field magnitude'  |  positive float  |

      where 'theta_z' and 'phi_z' are the polar and azimuthal angles of the magnetic field with respect to the LAB system (to be measured in radians), while field magnitude is to be expressed in tesla.
      
      When it is None, the Zeeman interaction is not taken into account.
      Default value is None.
    
    - j_matrix: np.ndarray
    
      Array whose elements represent the coefficients Jmn which determine the strength of the J-coupling between each pair of spins in the system. For the details on these data, see the description of the same-named parameter in the docstrings of the function h_j_coupling in the module Hamiltonians.py.
      
      When it is None, the J-coupling effects are not taken into account.      
      Default value is None.
    
    - initial_state: either string or numpy.ndarray
  
      Specifies the state of the system at time t=0.
    
      If the keyword canonical is passed, the function will return a Density_Matrix object representing the state of thermal equilibrium at the temperature specified by the same-named argument.
    
      If a square complex array is passed, the function will return a Density_Matrix object directly initialised with it.
    
      Default value is 'canonical'.
    
    - temperature: float
  
      Temperature of the system (in kelvin).
    
      Default value is 1e-4.
    
    Returns
    -------
    - [0]: Nuclear_Spin / Many_Spins
    
           The single spin/spin system subject to the NMR/NQR experiment.

    - [1]: Observable
  
           The unperturbed Hamiltonian, consisting of the Zeeman, quadrupolar and J-coupling terms (expressed in MHz).
    
    - [2]: Density_Matrix
  
           The density matrix representing the state of the system at time t=0, initialised according to initial_state.
    """
    
    if not isinstance(spin_par, list):
        spin_par = [spin_par]
    if quad_par is not None and not isinstance(quad_par, list):
        quad_par = [quad_par]
    
    if quad_par is not None and len(spin_par) != len(quad_par):
        raise IndexError("The number of passed sets of spin parameters must be equal to the number of the quadrupolar ones.")
    
    spins = []
    h_q = []
    h_z = []        
    
    h_unperturbed = 0
    
    for i in range(len(spin_par)):
        spins.append(Nuclear_Spin(spin_par[i]['quantum number'], \
                                  spin_par[i]['gamma/2pi']))
        
        if quad_par is not None:
            h_q.append(h_quadrupole(spins[i], quad_par[i]['coupling constant'], \
                                              quad_par[i]['asymmetry parameter'], \
                                              quad_par[i]['alpha_q'], \
                                              quad_par[i]['beta_q'], \
                                              quad_par[i]['gamma_q']))
        else:
            h_q.append(h_quadrupole(spins[i], 0., 0., 0., 0., 0.))
        
        if zeem_par is not None:
            h_z.append(h_zeeman(spins[i], zeem_par['theta_z'], \
                                          zeem_par['phi_z'], \
                                          zeem_par['field magnitude']))
        else:
            h_z.append(h_zeeman(spins[i], 0., 0., 0.))
    
    spin_system = Many_Spins(spins)
    
    h_unperturbed = Operator(spin_system.d)*0
    
    for i in range(spin_system.n_spins):
        h_i = h_q[i] + h_z[i]
        for j in range(i):
            h_i = tensor_product(Operator(spin_system.spin[j].d), h_i)
        for k in range(spin_system.n_spins)[i+1:]:
            h_i = tensor_product(h_i, Operator(spin_system.spin[k].d))
        h_unperturbed = h_unperturbed + h_i
    
    if j_matrix is not None:
        h_j = h_j_coupling(spin_system, j_matrix)
        h_unperturbed = h_unperturbed + h_j
    
    if isinstance(initial_state, str) and initial_state == 'canonical':
        dm_initial = canonical_density_matrix(h_unperturbed, temperature)
    else:
        dm_initial = Density_Matrix(initial_state)
    
    if len(spins) == 1:
        return spins[0], Observable(h_unperturbed.matrix), dm_initial
    else:
        return spin_system, Observable(h_unperturbed.matrix), dm_initial


def power_absorption_spectrum(spin, h_unperturbed, normalized=True, dm_initial=None):
    """
    Computes the spectrum of power absorption of the system due to x-polarized monochromatic pulses.
      
    Parameters
    ----------
    - spin: Nuclear_Spin / Many_Spins
  
            Single spin/spin system under study.
  
    - h_unperturbed: Operator
    
                     Unperturbed Hamiltonian of the system (in MHz).
    
    - normalized: bool
                
                  Specifies whether the difference between the states' populations are to be taken into account in the calculation of the line intensities. When normalized=True, they are not, when normalized=False, the intensities are weighted by the differences p(b)-p(a) just like in the formula above.
    
                  Default value is True.
  
    - dm_initial: Density_Matrix or None
  
                  Density matrix of the system at time t=0, just before the application of the pulse.
    
                  The default value is None, and it should be left so only when normalized=True, since the initial density matrix is not needed.
  
    Action
    ------
    Diagonalises h_unperturbed and computes the frequencies of transitions between its eigenstates.
  
    Then, it determines the relative proportions of the power absorption for different lines applying the formula derived from Fermi golden rule (taking or not taking into account the states' populations, according to the value of normalized).
  
    Returns
    -------
    [0]: The list of the frequencies of transition between the eigenstates of h_unperturbed (in MHz);
    
    [1]: The list of the corresponding intensities (in arbitrary units).
    """
    energies, o_change_of_basis = h_unperturbed.diagonalisation()
    
    transition_frequency = []
    
    transition_intensity = []
    
    d = h_unperturbed.dimension()
    
    # Operator of the magnetic moment of the spin system
    if isinstance(spin,  Many_Spins):
        magnetic_moment = Operator(spin.d)*0
        for i in range(spin.n_spins):
            mm_i = spin.spin[i].gyro_ratio_over_2pi*spin.spin[i].I['x']
            for j in range(i):
                mm_i = tensor_product(Operator(spin.spin[j].d), mm_i)
            for k in range(spin.n_spins)[i+1:]:
                mm_i = tensor_product(mm_i, Operator(spin.spin[k].d))
            magnetic_moment = magnetic_moment + mm_i
    else:
        magnetic_moment = spin.gyro_ratio_over_2pi*spin.I['x']
    
    mm_in_basis_of_eigenstates = magnetic_moment.sim_trans(o_change_of_basis)
    
    for i in range(d):
        for j in range(d):
            if i < j:
                nu = np.absolute(energies[j] - energies[i])
                transition_frequency.append(nu)
                
                intensity_nu = nu*\
                    (np.absolute(mm_in_basis_of_eigenstates.matrix[j, i]))**2
                
                if not normalized:
                    p_i = dm_initial.matrix[i, i]
                    p_j = dm_initial.matrix[j, j]
                    intensity_nu = np.absolute(p_i-p_j)*intensity_nu
                    
                transition_intensity.append(intensity_nu)
            else:
                pass
    
    return transition_frequency, transition_intensity


def plot_power_absorption_spectrum(frequencies, intensities, show=True, save=False, name='PowerAbsorptionSpectrum', destination=''):
    """
    Plots the power absorption intensities as a function of the corresponding frequencies.
  
    Parameters
    ----------
    - frequencies: array-like
                
                   Frequencies of the transitions (in MHz).
    
    - intensities: array-like
    
                   Intensities of the transitions (in a.u.).
    
    - show: bool
  
            When False, the graph constructed by the function will not be displayed.
    
            Default value is True.
    
    - save: bool
  
            When False, the plotted graph will not be saved on disk. When True, it will be saved with the name passed as name and in the directory passed as destination.
    
            Default value is False.
    
    - name: string
  
            Name with which the graph will be saved.
    
            Default value is 'PowerAbsorptionSpectrum'.
    
    - destination: string
  
                   Path of the directory where the graph will be saved (starting from the current directory). The name of the directory must be terminated with a slash /.
    
                   Default value is the empty string (current directory).
    
    Action
    ------
    If show=True, generates a graph with the frequencies of transition on the x axis and the corresponding intensities on the y axis.
    
    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure built up by the function.
    """
    fig = plt.figure()
    
    plt.vlines(frequencies, 0, intensities, colors='b')
    
    plt.xlabel("\N{GREEK SMALL LETTER NU} (MHz)")    
    plt.ylabel("Power absorption (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    if show: plt.show()
    
    return fig


def evolve(spin, h_unperturbed, dm_initial, \
           mode=None, pulse_time=0, \
           picture='RRF', RRF_par={'nu_RRF': 0,
                                   'theta_RRF': 0,
                                   'phi_RRF': 0}, \
           n_points=10, order=2):
    
    """
    Simulates the evolution of the density matrix of a nuclear spin under the action of an electromagnetic pulse in a NMR/NQR experiment.
  
    Parameters
    ----------
    - spin: Nuclear_Spin
  
            Spin under study.
    
    - h_unperturbed: Operator
  
                     Hamiltonian of the nucleus at equilibrium (in MHz).
    
    - dm_initial: Density_Matrix
  
                  Density matrix of the system at time t=0, just before the application of the pulse.

    - mode: pandas.DataFrame
  
            Table of the parameters of each electromagnetic mode in the pulse. See the description of the same-named argument of the function h_multiple_mode_pulse in page Hamiltonians for the details on the tabular organisation of these data.
            
            When it is None, the evolution of the system is performed for the given time duration without any applied pulse.
            
            The default value is None.
    
    - pulse_time: float
  
                  Duration of the pulse of radiation sent onto the sample (in microseconds).
                  
                  The default value is 0.
    
    - picture: string
  
               Sets the dynamical picture where the density matrix of the system is evolved. May take the values:
        1. IP', which sets the interaction picture;
        2.'RRF' (or anything else), which sets the picture corresponding to a rotating reference frame whose features are specified in argument RRF_par.
    
               The default value is RRF.
    
    - RRF_par: dict
  
               Specifies the properties of the rotating reference frame where evolution is carried out when picture='RRF'. The details on the organisation of these data can be found in the description of function RRF_Operator.
    
               By default, all the values in this map are set to 0 (RRF equivalent to the LAB frame).
    
    - n_points: float
  
                Counts the number of points in which the time interval [0, pulse_time] is sampled in the discrete approximation of the time-dependent Hamiltonian of the system.
    
                Default value is 10.
    
    - order : int
  
              Specifies at which order the Magnus expansion of the Hamiltonian is to be truncated. Anyway, for all the values greater than 3 the program will take into account only the 1st, 2nd and 3rd-order terms.
    
              Default value is 2.
  
    Action
    ------
    If
    - pulse_time is equal to 0;
    - dm_initial is very close to the identity (with an error margin of 1e-10 for each element)
  
    the function returns dm_initial without performing any evolution.
  
    Otherwise, evolution is carried out in the picture determined by the same-named parameter. The evolution operator is built up appealing to the Magnus expansion of the full Hamiltonian of the system (truncated to the order specified by the same-named argument).
  
    Returns
    -------
    The Density_Matrix object representing the state of the system (in the Schroedinger picture) evolved through a time pulse_time under the action of the specified pulse.
    """
    
    if pulse_time == 0 or np.all(np.absolute((dm_initial-Operator(spin.d)).matrix)<1e-10):
        return dm_initial
    
    if picture == 'IP':
        o_change_of_picture = h_unperturbed
    else:
        o_change_of_picture = RRF_operator(spin, RRF_par)
        
    if mode is None:
        mode = pd.DataFrame([(0., 0., 0., 0., 0)], 
                            columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
        
    times, time_step = np.linspace(0, pulse_time, num=max(2, int(n_points)), retstep=True)
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
def RRF_operator(spin, RRF_par):
    """
    Returns the operator for the change of picture equivalent to moving to the RRF.
  
    Parameters
    ----------
    - spin: Nuclear_Spin
            
            Spin under study.
  
    - RRF_par: dict
                
               Specifies the properties of the rotating reference frame. The keys and values required to this argument are shown in the table below:
    
               |      key      |  value  |
               |      ---      |  -----  |
               |    'nu_RRF'   |  float  |
               |  'theta_RRF'  |  float  |
               |   'phi_RRF'   |  float  |
    
    where 'nu_RRF' is the frequency of rotation of the RRF (in MHz), while 'theta_RRF' and 'phi_RRF' are the polar and azimuthal angles of the normal to the plane of rotation in the LAB frame (in radians).
    
    Returns
    -------
    An Observable object representing the operator which generates the change to the RRF picture.
    """
    nu = RRF_par['nu_RRF']
    theta = RRF_par['theta_RRF']
    phi = RRF_par['phi_RRF']
    RRF_o = nu*(spin.I['z']*math.cos(theta) + \
                spin.I['x']*math.sin(theta)*math.cos(phi) + \
                spin.I['y']*math.sin(theta)*math.sin(phi))
    return Observable(RRF_o.matrix)


def plot_real_part_density_matrix(dm, many_spin_indexing = None, show=True, save=False, name='RealPartDensityMatrix', destination=''):
    """
    Generates a 3D histogram displaying the real part of the elements of the passed density matrix.
  
    Parameters
    ----------
    - dm: Density_Matrix
  
          Density matrix to be plotted.
          
    - many_spin_indexing: either None or list
  
                          If it is different from None, the density matrix dm is interpreted as the state of a many spins' system, and this parameter provides the list of the dimensions of the subspaces of the full Hilbert space related to the individual nuclei of the system. The ordering of the elements of many_spin_indexing should match that of the single spins' density matrices in their tensor product resulting in dm.
                          Default value is None.
    
    - show: bool
  
            When False, the graph constructed by the function will not be displayed.
    
            Default value is True.
    
    - save: bool
  
            When False, the plotted graph will not be saved on disk. When True, it will be saved with the name passed as name and in the directory passed as destination.
    
            Default value is False.
    
    - name: string
  
            Name with which the graph will be saved.
    
            Default value is 'RealPartDensityMatrix'.
    
    - destination: string
  
                   Path of the directory where the graph will be saved (starting from the current directory). The name of the directory must be terminated with a slash /.
    
                   Default value is the empty string (current directory).

    Action
    ------
    If show=True, draws a histogram on a 2-dimensional grid representing the density matrix, with the real part of each element indicated along the z axis. Blue bars indicate the positive matrix elements, red bars indicate the negative elements in absolute value.

    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure built up by the function.

    """    
    real_part = np.vectorize(np.real)
    data_array = real_part(dm.matrix)
    
    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data
    # You can think of this as the floor of the plot
    x_data, y_data = np.meshgrid(np.arange(data_array.shape[1])+0.25,
                                 np.arange(data_array.shape[0])+0.25)
    
    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays: x_data, y_data, z_data. The following
    # call boils down to picking one entry from each array and plotting a bar from (x_data[i],
    # y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
        
    bar_color = np.zeros(len(z_data), dtype=object)
        
    for i in range(len(z_data)):
        if z_data[i]<-1e-10:
            bar_color[i] = 'tab:red'
        else:
            bar_color[i] = 'tab:blue'
                            
    ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             dx, dy, np.absolute(z_data), color=bar_color)
    
    d = data_array.shape[0]
    tick_label = []
    
    if many_spin_indexing is None:
        for i in range(d):
            tick_label.append('|' + str(Fraction((d-1)/2-i)) + '>')

    else:        
        d_sub = many_spin_indexing
        n_sub = len(d_sub)
        m_dict = []
        
        for i in range(n_sub):
            m_dict.append({})
            for j in range(d_sub[i]):
                m_dict[i][j] = str(Fraction((d_sub[i]-1)/2-j))
        
        for i in range(d):
            tick_label.append('>')

        for i in range(n_sub)[::-1]:
            d_downhill = int(np.prod(d_sub[i+1:n_sub]))
            d_uphill = int(np.prod(d_sub[0:i]))
            
            for j in range(d_uphill):
                for k in range(d_sub[i]):
                    for l in range(d_downhill):
                        tick_label[j*d_sub[i]*d_downhill + k*d_downhill + l] = m_dict[i][k] + \
                            ', ' + tick_label[j*d_sub[i]*d_downhill + k*d_downhill + l]
        
        for i in range(d):
            tick_label[i] = '|' + tick_label[i]
            
        ax.tick_params(axis='both', which='major', labelsize=6)
            
    xticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), tick_label)
    yticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), tick_label)
    
    ax.set_zlabel("Re(\N{GREEK SMALL LETTER RHO})")
    
    
    legend_elements = [Patch(facecolor='tab:blue', label='<m|\N{GREEK SMALL LETTER RHO}|m> > 0'), \
                       Patch(facecolor='tab:red', label='<m|\N{GREEK SMALL LETTER RHO}|m> < 0')]
    
    ax.legend(handles=legend_elements, loc='upper left')
    
    if save:
        plt.savefig(destination + name)
    
    if show:
        plt.show()
        
    return fig


def FID_signal(spin, h_unperturbed, dm, acquisition_time, T2=100, theta=0, phi=0, reference_frequency=0, n_points=10):
    """ 
    Simulates the free induction decay signal (FID) measured after the shut-off of the electromagnetic pulse, once the evolved density matrix of the system, the time interval of acquisition, the relaxation time T2 and the direction of the detection coils are given.
  
    Parameters
    ----------
    - spin: Nuclear_Spin
    
            Spin under study.
    
    - h_unperturbed: Operator
  
                     Unperturbed Hamiltonian of the system (in MHz).
    
    - dm: Density_Matrix
  
          Density matrix representing the state of the system at the beginning of the acquisition of the signal.
    
    - acquisition_time: float
  
                        Duration of the acquisition of the signal, expressed in microseconds.
    
    - T2: float
  
          Characteristic time of relaxation of the component of the magnetization on the plane of detectionvanishing. It is measured in microseconds.
    
          Default value is 100 (microseconds).
    
    - theta, phi: float
  
                  Polar and azimuthal angles which specify the normal to the plane of detection of the FID signal (in radians).
    
                  Default values are theta=0, phi=0.
                  
    - reference_frequency: float
    
                           Specifies the frequency of rotation of the measurement apparatus with respect to the LAB system.
                           Default value is 0.
    
    - n_points: float
  
                Counts (approximatively) the number of points per microsecond in which the time interval [0, acquisition_time] is sampled for the generation of the FID signal.
    
                Default value is 10.
    
    Action
    ------
    Samples the time interval [0, acquisition_time] with n_points points per microsecond.
    
    The FID signal is simulated under the assumption that it is directly related to the time-dependent component of the magnetization on the plane specified by (theta, phi) of the LAB system.
    
    Returns
    -------
    [0]: numpy.ndarray
  
         Vector of equally spaced sampled instants of time in the interval [0, acquisition_time] (in microseconds).
  
    [1]: numpy.ndarray
  
         FID signal evaluated at the discrete times reported in the first output (in arbitrary units).
    """
    times = np.linspace(start=0, stop=acquisition_time, num=int(acquisition_time*n_points))
    
    FID = []
    
    # Computes the FID assuming that the detection coils record the time-dependence of the
    # magnetization on the plane perpendicular to (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    Iz = spin.I['z']
    Iy = spin.I['y']
    I_plus_rotated = (1j*phi*Iz).exp()*(1j*theta*Iy).exp()*spin.I['+']*(-1j*theta*Iy).exp()*(-1j*phi*Iz).exp()
    for t in times:
        dm_t = dm.free_evolution(h_unperturbed, t)
        FID.append((dm_t*I_plus_rotated*np.exp(-t/T2)*np.exp(-1j*2*math.pi*reference_frequency*t)).trace())
    
    return times, np.array(FID)


def plot_real_part_FID_signal(times, FID, show=True, save=False, name='FIDSignal', destination=''):
    """
    Plots the real part of the FID signal as a function of time.
  
    Parameters
    ----------
    - time: array-like
  
            Sampled instants of time (in microseconds).
    
    - FID: array-like
  
           Sampled FID values (in arbitrary units).
    
    - show: bool
  
            When False, the graph constructed by the function will not be displayed.
    
            Default value is True.
    
    - save: bool
  
            When False, the plotted graph will not be saved on disk. When True, it will be saved with the name passed as name and in the directory passed as destination.
    
            Default value is False.
    
    - name: string
  
            Name with which the graph will be saved.
    
            Default value is 'FIDSignal'.
    
    - destination: string
  
                   Path of the directory where the graph will be saved (starting from the current directory). The name of the directory must be terminated with a slash /.
    
                   Default value is the empty string (current directory).
    
    Action
    ------ 
    If show=True, generates a plot of the FID signal as a function of time.
      
    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure built up by the function.
    """
    fig = plt.figure()
    
    plt.plot(times, np.real(FID), label='Real part')
        
    plt.xlabel("time (\N{GREEK SMALL LETTER MU}s)")    
    plt.ylabel("Re(FID) (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    if show: plt.show()
    
    return fig


def fourier_transform_signal(times, signal, frequency_start, frequency_stop, opposite_frequency=False):
    """
    Computes the Fourier transform of the passed time-dependent signal over the frequency interval [frequency_start, frequency_stop]. The implemented Fourier transform operation is
    
    where S is the original signal and T is its duration. In order to have a reliable Fourier transform, the signal should be very small beyond time T.

    Parameters
    ----------
    - times: array-like
  
             Sampled time domain (in microseconds).
    
    - signal: array-like
  
              Sampled signal to be transformed in the frequency domain (in a.u.).
  
    - frequency_start, frequency_stop: float
  
                                       Left and right bounds of the frequency interval of interest, respectively (in MHz).
    
    - opposite_frequency: bool
  
                          When it is True, the function computes the Fourier spectrum of the signal in both the intervals frequency_start -> frequency_stop and -frequency_start -> -frequency_stop (the arrow specifies the ordering of the Fourier transform's values when they are stored in the arrays to be returned).
    
    Returns
    -------
    [0]: numpy.ndarray
  
         Vector of 1000 equally spaced sampled values of frequency in the interval [frequency_start, frequency_stop] (in MHz).

    [1]: numpy.ndarray
  
         Fourier transform of the signal evaluated at the discrete frequencies reported in the first output (in a.u.).
    
    If opposite_frequency=True, the function also returns:
    
    [2]: numpy.ndarray
  
         Fourier transform of the signal evaluated at the discrete frequencies reported in the first output changed by sign (in a.u.).
    """
    dt = times[1]-times[0]
    
    frequencies = np.linspace(start=frequency_start, stop=frequency_stop, num=1000)
    
    fourier = [[], []]
    
    if opposite_frequency == False:
        sign_options = 1
    else:
        sign_options = 2
    
    for s in range(sign_options):
        for nu in frequencies:
            integral = np.zeros(sign_options, dtype=complex)
            for t in range(len(times)):
                integral[s] = integral[s] + np.exp(-1j*2*math.pi*(1-2*s)*nu*times[t])*signal[t]*dt
            fourier[s].append(integral[s])
        
    if opposite_frequency == False:
        return frequencies, np.array(fourier[0])
    else:
        return frequencies, np.array(fourier[0]), np.array(fourier[1])


# Finds out the phase responsible for the displacement of the real and imaginary parts of the Fourier
# spectrum of the FID with respect to the ideal absorptive/dispersive lorentzian shapes
def fourier_phase_shift(frequencies, fourier, fourier_neg=None, peak_frequency=0, int_domain_width=0.5):
    """
    Computes the phase factor which must multiply the Fourier spectrum (`fourier`) in order to have the real and imaginary part of the adjusted spectrum showing the conventional dispersive/absorptive shapes at the peak specified by `peak_frequency`.

    Parameters
    ----------
    - frequencies: array-like
  
                   Sampled values of frequency (in MHz). 
  
    - fourier: array-like
  
               Values of the Fourier transform of the signal (in a.u.) sampled at the frequencies passed as the first argument.
    
    - fourier_neg: array-like
  
                   Values of the Fourier transform of the signal (in a.u.) sampled at the opposite of the frequencies passed as the first argument. When fourier_neg is passed, it is possible to specify a peak_frequency located in the range frequencies changed by sign.
    
                   Default value is None.
  
    - peak_frequency: float
  
                      Position of the peak of interest in the Fourier spectrum.
    
                      Default value is 0.

    - int_domain_width: float
  
                        Width of the domain (centered at peak_frequency) where the real and imaginary parts of the Fourier spectrum will be integrated.
    
                        Default value is 0.5.

    Action
    ------  
    The function integrates both the real and the imaginary parts of the spectrum over an interval of frequencies centered at peak_frequency whose width is given by int_domain_width. Then, it computes the phase shift.
    
    Returns
    -------
    A float representing the desired phase shift (in radians).
    """
    
    if fourier_neg is not None:
        fourier = np.concatenate((fourier, fourier_neg))
        frequencies = np.concatenate((frequencies, -frequencies))
    
    integration_domain = np.nonzero(np.isclose(frequencies, peak_frequency, atol=int_domain_width/2))[0]
    
    int_real_fourier = 0
    int_imag_fourier = 0
    
    for i in integration_domain:
        int_real_fourier = int_real_fourier + np.real(fourier[i])
        int_imag_fourier = int_imag_fourier + np.imag(fourier[i])
        
    if np.absolute(int_real_fourier) < 1e-10 :
        if int_imag_fourier > 0:
            return 0
        else:
            return math.pi
    
    atan = math.atan(-int_imag_fourier/int_real_fourier)
    
    if int_real_fourier > 0:
        phase = atan + math.pi/2
    else:
        phase = atan - math.pi/2
        
    return phase


# If another set of data is passed as fourier_neg, the function plots a couple of graphs, with the
# one at the top interpreted as the NMR signal produced by a magnetization rotating counter-clockwise,
# the one at the bottom corresponding to the opposite sense of rotation
def plot_fourier_transform(frequencies, fourier, fourier_neg=None, square_modulus=False, 
                           scaling_factor=None, show=True, save=False, name='FTSignal', destination=''):
    """
    Plots the Fourier transform of a signal as a function of the frequency.
  
    Parameters
    ----------
    - frequencies: array-like
  
                   Sampled values of frequency (in MHz).
    
    - fourier: array-like
  
               Sampled values of the Fourier transform (in a.u.).
    
    - fourier_neg: array-like
  
                   Sampled values of the Fourier transform (in a.u.) evaluated at the frequencies in frequencies changed by sign.
    
                   Default value is None.
    
    - square_modulus: bool
  
                      When True, makes the function plot the square modulus of the Fourier spectrum rather than the separate real and imaginary parts, which is the default option (by default, square_modulus=False).
                      
    - scaling_factor: float
    
                      When it is not None, it specifies the scaling factor which multiplies the data to be plotted. It applies simultaneously to all the plots in the resulting figure.
    
    - show: bool
  
            When False, the graph constructed by the function will not be displayed.
    
            Default value is True.
    
    - save: bool
  
            When False, the plotted graph will not be saved on disk. When True, it will be saved with the name passed as name and in the directory passed as destination.
    
            Default value is False.
    
    - name: string
  
            Name with which the graph will be saved.
    
            Default value is 'FTSignal'.
    
    - destination: string
  
                   Path of the directory where the graph will be saved (starting from the current directory). The name of the directory must be terminated with a slash /.
    
                   Default value is the empty string (current directory).
    
    Action
    ------
    Builds up a plot of the Fourier transform of the passed complex signal as a function of the frequency.
    If fourier_neg is different from None, two graphs are built up which represent respectively the Fourier spectra for counter-clockwise and clockwise rotation frequencies.
    
    If show=True, the figure is printed on screen.  
  
    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure built up by the function.
    """
    if fourier_neg is None:
        n_plots = 1
        fourier_data = [fourier]
    else:
        n_plots = 2
        fourier_data = [fourier, fourier_neg]
        plot_title = ["Counter-clockwise precession", "Clockwise precession"]
    
    fig, ax = plt.subplots(n_plots, 1, sharey=True, gridspec_kw={'hspace':0.5})
    
    if fourier_neg is None:
        ax = [ax]
        
    if scaling_factor is not None:
        for i in range(n_plots):
            fourier_data[i] = scaling_factor*fourier_data[i]
        
    for i in range(n_plots):
        if not square_modulus:
            ax[i].plot(frequencies, np.real(fourier_data[i]), label='Real part')
            ax[i].plot(frequencies, np.imag(fourier_data[i]), label='Imaginary part')
        else:
            ax[i].plot(frequencies, np.absolute(fourier_data[i])**2, label='Square modulus')
        
        if n_plots>1:
            ax[i].title.set_text(plot_title[i])
        
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel("frequency (MHz)")    
        ax[i].set_ylabel("FT signal (a. u.)")  
         
    if save: plt.savefig(destination + name)
        
    if show: plt.show()
        
    return fig






