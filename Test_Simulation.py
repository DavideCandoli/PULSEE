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

# Checks if the simulation works, doesn't check wrong output
def test_Generic_Simulation():
    mode = pd.DataFrame([(5., 10., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta', 'phi'])
    Simulate(1, 1., \
             0, 0, 5., \
             2., 2., 0.5, math.pi/2, math.pi/2, math.pi/2, \
             300, \
             mode, \
             5.)