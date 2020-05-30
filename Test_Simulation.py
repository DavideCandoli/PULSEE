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
    mode = pd.DataFrame([(5., 10., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta', 'phi'])
    Simulate(s=1, gyro_ratio=1., \
             theta_z=0, phi_z=0, H_0=5., \
             eQ=2., eq=2., eta=0.5, alpha_q=math.pi/2, beta_q=math.pi/2, gamma_q=math.pi/2, \
             temperature=300, \
             mode=mode, \
             pulse_duration=5.)