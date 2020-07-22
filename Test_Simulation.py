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

from Hamiltonians import H_Zeeman, H_Quadrupole, H_Changed_Picture

from Simulation import Nuclear_System_Setup, Evolve

# Checks if the simulation works, doesn't check if there is wrong output
def test_Generic_Simulation():
    spin_par = {'quantum number' : 1,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 5.,
                'theta_z' : math.pi/2,
                'phi_z' : math.pi/2}
    quad_par = {'coupling constant' : 2.,
                'asymmetry parameter' : 0.5,
                'alpha_q' : math.pi,
                'beta_q' : math.pi/2,
                'gamma_q' : 0}
    mode = pd.DataFrame([(5., 10., 0., math.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    spin, h_unperturbed, dm_0 = Nuclear_System_Setup(spin_par, zeem_par, quad_par, initial_state='canonical')
    
    Evolve(spin, h_unperturbed, dm_0, \
           mode, pulse_time=5.)