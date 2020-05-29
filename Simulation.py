import numpy as np
import math

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term

from Nuclear_Spin import Nuclear_Spin

from Hamiltonians import H_Zeeman, H_Quadrupole, \
                         H_Pulse, V0, V1, V2


# This instance of the Nuclear_Spin class is the basic object underpinning all the calculations in the following simulation, so it is introduced as a global variable
#spin = Nuclear_Spin()

# Main function of the program, it runs every step of the simulation
#def Simulate():
#    spin.set_quantum_number(Input_Spin_Quantum_Number())
#    pulse_time = Input_Pulse_Time()
#    initial_density_matrix = Compute_Canonical_Density_Matrix()
#    evolved_density_matrix = Evolve(initial_density_matrix, pulse_time)
#    return evolved_density_matrix

# Drives the evolution of the state of the ensemble of nuclear spins
#def Evolve(density_matrix_0, T):
#    h1_IP_Magnus0th = Compute_Magnus_0th_Term(H1_IP, T)
#    h1_IP_Magnus1st = Compute_Magnus_1st_Term(H1_IP, T)
#    h1_IP_average = h1_IP_Magnus0th + h1_IP_Magnus1st
#    density_matrix_IP_T = Unitary_Sandwich(density_matrix_0, -h1_IP_average*T)
#    density_matrix_T = Interaction_Picture(density_matrix_IP_T, H0, T, invert=True)
#    return density_matrix_T

# Asks for the spin quantum number as an input from terminal
#def Input_Spin_Quantum_Number():
#    return 1.

# Asks for the duration of the pulse as an input from terminal
#def Input_Pulse_Time():
#    return 1.

# Calculates the expression of the density matrix associated to a state of the ensemble of nuclear spins at thermal equilibrium
#def Compute_Canonical_Density_Matrix():
#    canonical_density_matrix = np.zeros((spin.d, spin.d), dtype=complex)
#    for m in range(spin.d):
#        canonical_density_matrix[m][m] = 1/(spin.d)
#    return canonical_density_matrix

# Calculates the 0th term of Magnus expansion of the given Hamiltonian
#def Compute_Magnus_0th_Term(Hamiltonian, time_interval):
#    magnus0th = np.zeros((spin.d, spin.d))
#    return magnus0th

# Calculates the 1st term of Magnus expansion of the given Hamiltonian
#def Compute_Magnus_1st_Term(Hamiltonian, time_interval):
#    magnus1st = np.zeros((spin.d, spin.d))
#    return magnus1st

# Performs the matrix product U x M x U^(-1), where M stands for matrix and U is the complex exponential of exponent_operator
#def Unitary_Sandwich(matrix, exponent_operator):
#    return matrix

# Casts the input operator matrix into the interaction picture with respect to the Unperturbed_Hamiltonian at instant time. When invert is true, the function performs the inverse operation (receives the matrix in the interaction picture and returns its form in the standard one)
#def Interaction_Picture(matrix, Unperturbed_Hamiltonian, time=1., invert=False):
#    return matrix

# Stationary Hamiltonian of the system
#def H0():
#    return H_quadrupole + H_Zeeman

# Quadrupolar interaction term of the Hamiltonian
#def H_quadrupole():
#    h_quadrupole = np.zeros((spin.d, spin.d))
#    return h_quadrupole

# Zeeman term of the Hamiltonian 
#def H_Zeeman():
#    h_Zeeman = np.zeros((spin.d, spin.d))
#    return h_Zeeman

# Perturbation Hamiltonian (electromagnetic pulse)
#def H1(time):
#    h1 = np.zeros((spin.d, spin.d))
#    return h1

# Perturbation Hamiltonian cast in the interaction picture
#def H1_IP(time):
#    Interaction_Picture(H1(time), H0, time)
    
    