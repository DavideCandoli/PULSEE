import numpy as np
import math
from math import isnan
from grappa import should, expect

# Objects of the class Nuclear_Spin include all the mathematical tools connected with the observables of nuclear spin in quantum mechanics
class Nuclear_Spin:
    def __init__(self, s:int=1.):
        self.quantum_number=s
        # Shorthand notation for the spin multiplicity ('d' stands for the dimensionality of the spin's Hilbert space)
        self.d = self.multiplicity()
        # Shorthand notation for the spin operators in cartesian and spherical coordinates
        self.I = {1: self.cartesian_operator()[0],
                  2: self.cartesian_operator()[1],
                  3: self.cartesian_operator()[2],
                  '+': self.lowering_operator(),
                  '-': self.raising_operator()}
    # This function has been defined as a replacement of the conventional __setattr__ method, because updating the value of the variable quantum_number requires the attributes d and I to be updated accordingly
    def set_quantum_number(self, s:int):
        self.quantum_number=s
        self.d = self.multiplicity()
        self.I = {1: self.cartesian_operator()[0],
                  2: self.cartesian_operator()[1],
                  3: self.cartesian_operator()[2],
                  '+': self.lowering_operator(),
                  '-': self.raising_operator()}
    def multiplicity(self):
        return int(2*self.quantum_number+1)
    # Raising and lowering operator are initialized through the direct insertion of their matrix elements in the basis of eigenvectors of Iz
    def raising_operator(self):
        I_raising = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == 1:
                    I_raising[m, n] = math.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n + 1))
        return I_raising
    def lowering_operator(self):
        I_lowering = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == -1:
                    I_lowering[m, n] = math.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n - 1))
        return I_lowering
    # The two transversal components of the spin vector operator (I[0] and I[1] in the following scope) are constructed starting from their relations with the ladder operators, while the third component (I[2]) is initialized by direct insertion of its eigenvalues
    def cartesian_operator(self):
        I = [np.zeros((self.d, self.d)), np.zeros((self.d, self.d)), np.zeros((self.d, self.d))]
        I[0] = (self.raising_operator() + self.lowering_operator())/2
        I[1] = (self.raising_operator() - self.lowering_operator())/(2j)
        for m in range(self.d):
            I[2][m, m] = self.quantum_number - m
        return I

# This instance of the Nuclear_Spin class is the basic object underpinning all the calculations in the following simulation, so it is introduced as a global variable
spin = Nuclear_Spin()

# Main function of the program, it runs every step of the simulation
def Simulate():
    spin.set_quantum_number(Input_Spin_Quantum_Number())
    pulse_time = Input_Pulse_Time()
    initial_density_matrix = Compute_Canonical_Density_Matrix()
    evolved_density_matrix = Evolve(initial_density_matrix, pulse_time)
    return evolved_density_matrix

# Drives the evolution of the state of the ensemble of nuclear spins
def Evolve(density_matrix_0, T):
    h1_IP_Magnus0th = Compute_Magnus_0th_Term(H1_IP, T)
    h1_IP_Magnus1st = Compute_Magnus_1st_Term(H1_IP, T)
    h1_IP_average = h1_IP_Magnus0th + h1_IP_Magnus1st
    density_matrix_IP_T = Unitary_Sandwich(density_matrix_0, -h1_IP_average*T)
    density_matrix_T = Interaction_Picture(density_matrix_IP_T, H0, T, invert=True)
    return density_matrix_T

# Asks for the spin quantum number as an input from terminal
def Input_Spin_Quantum_Number():
    return 1.

# Asks for the duration of the pulse as an input from terminal
def Input_Pulse_Time():
    return 1.

# Calculates the expression of the density matrix associated to a state of the ensemble of nuclear spins at thermal equilibrium
def Compute_Canonical_Density_Matrix():
    canonical_density_matrix = np.zeros((spin.d, spin.d), dtype=complex)
    for m in range(spin.d):
        canonical_density_matrix[m][m] = 1/(spin.d)
    return canonical_density_matrix

def Trace(matrix):
    try:
        len(matrix.shape) | should.be.equal.to(2)
    except AssertionError as wrongshape:
        print(wrongshape)
    try:
        matrix.shape[0] | should.be.equal.to(matrix.shape[1])
    except AssertionError as notsquare:
        print(notsquare)
    trace = 0
    for m in range(len(matrix)):
        trace = trace + matrix[m][m]
    return trace

# Calculates the 0th term of Magnus expansion of the given Hamiltonian
def Compute_Magnus_0th_Term(Hamiltonian, time_interval):
    magnus0th = np.zeros((spin.d, spin.d))
    return magnus0th

# Calculates the 1st term of Magnus expansion of the given Hamiltonian
def Compute_Magnus_1st_Term(Hamiltonian, time_interval):
    magnus1st = np.zeros((spin.d, spin.d))
    return magnus1st

# Performs the matrix product U x M x U^(-1), where M stands for matrix and U is the complex exponential of exponent_operator
def Unitary_Sandwich(matrix, exponent_operator):
    return matrix

# Casts the input operator matrix into the interaction picture with respect to the Unperturbed_Hamiltonian at instant time. When invert is true, the function performs the inverse operation (receives the matrix in the interaction picture and returns its form in the standard one)
def Interaction_Picture(matrix, Unperturbed_Hamiltonian, time=1., invert=False):
    return matrix

# Stationary Hamiltonian of the system
def H0():
    return H_quadrupole + H_Zeeman

# Quadrupolar interaction term of the Hamiltonian
def H_quadrupole():
    h_quadrupole = np.zeros((spin.d, spin.d))
    return h_quadrupole

# Zeeman term of the Hamiltonian 
def H_Zeeman():
    h_Zeeman = np.zeros((spin.d, spin.d))
    return h_Zeeman

# Perturbation Hamiltonian (electromagnetic pulse)
def H1(time):
    h1 = np.zeros((spin.d, spin.d))
    return h1

# Perturbation Hamiltonian cast in the interaction picture
def H1_IP(time):
    Interaction_Picture(H1(time), H0, time)
    
    