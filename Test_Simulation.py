import numpy as np
import pandas as pd
import math

import hypothesis.strategies as st
from hypothesis import given, assume

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term

from Nuclear_Spin import Nuclear_Spin

from Hamiltonians import H_Zeeman, H_Quadrupole, H_Pulse_IP

from Simulation import Simulate

# Checks if the simulation works, doesn't check if there is wrong output
def test_Generic_Simulation():
    spin_par = pd.DataFrame([(1, 1.)],
                            columns=['quantum number', 'gyromagnetic ratio'])
    zeem_par = pd.DataFrame([(5., 0, 0)],
                            columns=['field magnitude', 'theta', 'phi'])
    quad_par = pd.DataFrame([(2., 0.5, math.pi/2, math.pi/2, math.pi/2)],
                            columns=['coupling constant', 'asymmetry parameter', 'alpha', 'beta', 'gamma'])
    mode = pd.DataFrame([(5., 10., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta', 'phi'])
    Simulate(spin_par, \
             zeem_par, \
             quad_par, \
             mode=mode, \
             temperature=300, \
             pulse_time=5.)