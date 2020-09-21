import math
import numpy as np

from Operators import *

# An instance of the following class is to be thought as an all-round representation of the nuclear spin
# angular momentum. Indeed, it includes all the operators typically associated with the spin and
# also specific parameters like the spin quantum number and the spin multiplicity
class Nuclear_Spin:

    def __init__(self, s=1, gamma_over_2pi=1):
        s = float(s)
        if not math.isclose(int(2*s), 2*s, rel_tol=1e-10):
            raise ValueError("The given spin quantum number is not a half-integer number")
        self.quantum_number = s
        self.d = self.multiplicity()
        self.I = {'+': self.lowering_operator(),
                  '-': self.raising_operator(),
                  'x': self.cartesian_operator()[0],
                  'y': self.cartesian_operator()[1],
                  'z': self.cartesian_operator()[2]}

        self.gyro_ratio_over_2pi = float(gamma_over_2pi)
    
    def multiplicity(self):
        return int((2*self.quantum_number)+1)

    def raising_operator(self):
        I_raising = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == 1:
                    I_raising[m, n] = math.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n + 1))
        return Operator(I_raising)

    def lowering_operator(self):
        I_lowering = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == -1:
                    I_lowering[m, n] = math.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n - 1))
        return Operator(I_lowering)

    def cartesian_operator(self):
        I = []
        I.append(Observable(((self.raising_operator() + self.lowering_operator())/2).matrix))
        I.append(Observable(((self.raising_operator() - self.lowering_operator())/(2j)).matrix))
        I.append(Observable(self.d))
        for m in range(self.d):
            I[2].matrix[m, m] = self.quantum_number - m
        return I