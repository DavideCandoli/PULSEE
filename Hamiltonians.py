import math
import numpy as np

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term

from Nuclear_Spin import Nuclear_Spin

# Returns the Observable object representing the Hamiltonian for the Zeeman interaction with an external
# static field
def H_Zeeman(spin, theta_z, phi_z, H_0):
    if H_0<0: raise ValueError("The modulus of the magnetic field must be a non-negative quantity")
    h_Zeeman = -spin.gyromagnetic_ratio*H_0* \
                (math.sin(theta_z)*math.cos(phi_z)*spin.I['x'] + \
                 math.sin(theta_z)*math.sin(phi_z)*spin.I['y'] + \
                 math.cos(theta_z)*spin.I['z'])
    return Observable(h_Zeeman.matrix)

def H_Quadrupole(spin, eQ, eq, eta, alpha, beta, gamma):
    if eta<0 or eta>1: raise ValueError("The asymmetry parameter must fall in the interval [0, 1]")
    h_quadrupole = (eQ/(spin.quantum_number*(2*spin.quantum_number-1)))* \
                   ((1/2)*(3*(spin.I['z']**2) - \
                           Operator(spin.d)*spin.quantum_number*(spin.quantum_number+1))* \
                    V0(eq, eta, alpha, beta, gamma) + \
                    (math.sqrt(6)/4)* \
                    ((spin.I['z']*spin.I['+'] + \
                      spin.I['+']*spin.I['z'])* \
                     V1(-1, eq, eta, alpha, beta, gamma) - \
                     (spin.I['z']*spin.I['-'] + \
                      spin.I['-']*spin.I['z'])* \
                     V1(+1, eq, eta, alpha, beta, gamma) + \
                     (spin.I['+']**2)* \
                      V2(-2, eq, eta, alpha, beta, gamma) + \
                     (spin.I['-']**2)* \
                      V2(2, eq, eta, alpha, beta, gamma)))
    return Observable(h_quadrupole.matrix)

def V0(eq, eta_q, alpha_q, beta_q, gamma_q):
    return 1.

def V1(sign, eq, eta, alpha, beta, gamma):
    return 1j

def V2(sign, eq, eta, alpha, beta, gamma):
    return 1j*sign

def H_Pulse():
    pass