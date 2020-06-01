import numpy as np
import pandas as pd
import math

from Simulation import Simulate


def Zeeman_Spectrum():
    spin_par = {'quantum number' : 1,
                'gyromagnetic ratio' : 1.}
    zeem_par = {'field magnitude' : 1.,
                'theta_z' : math.pi/2,
                'phi_z' : math.pi/2}
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0.,
                'beta_q' : 0.,
                'gamma_q' : 0}
    mode = pd.DataFrame([(0., 0., 0., 0., 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_z', 'phi_z'])
    Simulate(spin_par, \
             zeem_par, \
             quad_par, \
             mode=mode, \
             temperature=300, \
             pulse_time=0)