import math
from cmath import exp
import numpy as np
import pandas as pd

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


# Returns the Observable object representing the Hamiltonian for the interaction between the quadrupole
# moment of the nucleus and the electric field gradient
def H_Quadrupole(spin, eQ, eq, eta, alpha, beta, gamma):
    if eta<0 or eta>1: raise ValueError("The asymmetry parameter must fall in the interval [0, 1]")
    h_quadrupole = (eQ/(spin.quantum_number*(2*spin.quantum_number-1)))* \
                   ((1/2)*(3*(spin.I['z']**2) - \
                           Operator(spin.d)*spin.quantum_number*(spin.quantum_number+1))* \
                    V0(eq, eta, alpha, beta, gamma) + \
                    (math.sqrt(6)/4)* \
                    ((spin.I['z']*spin.I['+'] + \
                      spin.I['+']*spin.I['z'])* \
                     V1(-1, eq, eta, alpha, beta, gamma) + \
                     (spin.I['z']*spin.I['-'] + \
                      spin.I['-']*spin.I['z'])* \
                     V1(+1, eq, eta, alpha, beta, gamma) + \
                     (spin.I['+']**2)* \
                      V2(-2, eq, eta, alpha, beta, gamma) + \
                     (spin.I['-']**2)* \
                      V2(2, eq, eta, alpha, beta, gamma)))
    return Observable(h_quadrupole.matrix)


# Returns the spherical component V^0 of the EFG tensor
def V0(eq, eta, alpha, beta, gamma):
    v0 = (eq/2)*\
         (
          ((3*(math.cos(beta))**2-1)/2) - (eta*(math.sin(beta))**2)*(math.cos(2*gamma))/2
         )
    return v0


# Returns the spherical components V^{+/-1} of the EFG tensor
def V1(sign, eq, eta, alpha, beta, gamma):
    sign = np.sign(sign)
    v1 = (eq/2)*\
         (
          -sign*1j*math.sqrt(3/8)*math.sin(2*beta)*exp(sign*1j*alpha)+\
          1j*(eta/(math.sqrt(6)))*math.sin(beta)*\
          (
           -((1+sign*math.cos(beta))/2)*exp(1j*(sign*alpha+2*gamma))+\
            ((1-sign*math.cos(beta))/2)*exp(1j*(sign*alpha-2*gamma))
          )
         )
    return v1


# Returns the spherical components V^{+/-2} of the EFG tensor
def V2(sign, eq, eta, alpha, beta, gamma):
    sign = np.sign(sign)
    v2 = (eq/2)*\
         (-math.sqrt(3/8)*((math.sin(beta))**2)*exp(sign*2j*alpha) +\
          (eta/math.sqrt(6))*exp(sign*2j*alpha)*\
           (
            exp(2j*gamma)*((1+sign*math.cos(beta))**2)/4 +\
            exp(-2j*gamma)*((1-sign*math.cos(beta))**2)/4
           )
         )
    return v2


# Returns the Observable object representing the Hamiltonian of the interaction between the nucleus
# and a time-dependent, monochromatic and linearly polarized electromagnetic pulse
def H_Single_Mode_Pulse(spin, frequency, H_1, phase, theta, phi, t):
    if frequency < 0: raise ValueError("The angular frequency of the electromagnetic wave must be a positive quantity")
    if H_1 < 0: raise ValueError("The amplitude of the electromagnetic wave must be a positive quantity")
    h_pulse = -spin.gyromagnetic_ratio*H_1*\
              (math.sin(theta)*math.cos(phi)*spin.I['x'] +\
               math.sin(theta)*math.sin(phi)*spin.I['y'] +\
               math.cos(theta)*spin.I['z']
               )*\
               math.cos(frequency*t-phase)
    return Observable(h_pulse.matrix)
    

# Returns the Hamiltonian of interaction between the nucleus and multiple single-mode electromagnetic
# pulses
def H_Multiple_Mode_Pulse(spin, mode, t):
    h_pulse = Operator(spin.d)*0
    omega = mode['frequency']
    H = mode['amplitude']
    phase = mode['phase']
    theta = mode['theta']
    phi = mode['phi']
    for i in mode.index:
        h_pulse = h_pulse + H_Single_Mode_Pulse(spin, omega[i], H[i], phase[i], theta[i], phi[i], t)
    return Observable(h_pulse.matrix)


# Hamiltonian of the pulse cast into the interaction picture
def H_Pulse_IP(spin, mode, t, h_change_of_picture):
    h_pulse_ip = H_Multiple_Mode_Pulse(spin, mode, t).interaction_picture(h_change_of_picture, t)
    return Observable(h_pulse_ip.matrix)
    






