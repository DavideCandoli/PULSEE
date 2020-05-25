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
    h_quadrupole = eQ/(spin.quantum_number*(2*spin.quantum_number-1))* \
                   ((1/2)*(3*(spin.I['z']**2)-spin.quantum_number*(spin.quantum_number+1))* \
                    V0(eq, eta, alpha, beta, gamma) + \
                    (math.sqrt(6)/4)* \
                    ((spin.I['z']*spin.I['+'] + spin.I['+']*spin.I['z'])* \
                     V1(-1, eq, eta, alpha, beta, gamma) - \
                     (spin.I['z']*spin.I['-'] + spin.I['-']*spin.I['z'])* \
                     V1(+1, eq, eta, alpha, beta, gamma) + \
                     (spin.I['+']**2)*V2(-2, eq, eta, alpha, beta, gamma) + \
                     (spin.I['-']**2)*V2(+2, eq, eta, alpha, beta, gamma)))
    return h_quadrupole

def V0(eq, eta, alpha, beta, gamma):
    pass

def V1(sign, eq, eta, alpha, beta, gamma):
    pass

def V2(sign, eq, eta, alpha, beta, gamma):
    pass

def H_Pulse():
    pass