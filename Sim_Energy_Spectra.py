import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt

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

def Zeeman_Spectrum():
    spin = Nuclear_Spin(1, 1.)
    h_zeeman = H_Zeeman(spin, math.pi/2, math.pi/2, 1.)
    energy_spectrum = h_zeeman.eigenvalues()
    print("Energy spectrum of the pure Zeeman Hamiltonian = %r" % energy_spectrum)
    
    
    
    
    
    