import math
import numpy as np

from Operators import Operator, Density_Matrix, \
                      Observable, Random_Operator, \
                      Random_Observable, Random_Density_Matrix, \
                      Commutator, \
                      Magnus_Expansion_1st_Term, \
                      Magnus_Expansion_2nd_Term

# An instance of the following class is to be thought as an all-round representation of the nuclear spin
# angular momentum. Indeed, it includes all the Operators typically associated with the spin and also
# specific parameters like the spin quantum number and the spin multiplicity
class Nuclear_Spin:

    # The constructor of Nuclear_Spin receives as an argument only the spin quantum number s
    # and checks that this is a half-integer number as expected, raising appropriate errors if it
    # isn't. Then, all other attributes are initialised from the quantum number s.
    def __init__(self, s=1, gamma=1):
        try:
            s = float(s)
        except:
            raise TypeError("The given spin quantum number cannot be interpreted as a float")
        if not math.isclose(int(2*s), 2*s, rel_tol=1e-10):
            raise ValueError("The given spin quantum number is not a half-integer number")
        self.quantum_number = s
        self.d = self.multiplicity()
        self.I = {'+': self.lowering_operator(),
                  '-': self.raising_operator(),
                  'x': self.cartesian_operator()[0],
                  'y': self.cartesian_operator()[1],
                  'z': self.cartesian_operator()[2]}
        try:
            gamma = float(gamma)
        except:
            raise TypeError("The given gyromagnetic ratio cannot be interpreted as a float")
        self.gyromagnetic_ratio = gamma
    
    # Computes the dimensions of the spin Hilbert space
    def multiplicity(self):
        return int((2*self.quantum_number)+1)

    # Returns the spin raising Operator of dimensions given by multiplicity()
    def raising_operator(self):
        I_raising = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == 1:
                    I_raising[m, n] = math.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n + 1))
        return Operator(I_raising)

    # Returns the spin lowering Operator of dimensions given by multiplicity()
    def lowering_operator(self):
        I_lowering = np.zeros((self.d, self.d))
        for m in range(self.d):
            for n in range(self.d):
                if n - m == -1:
                    I_lowering[m, n] = math.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n - 1))
        return Operator(I_lowering)

    # Returns a list of Operator objects representing in the order the x, y and z cartesian 
    # components of the spin
    def cartesian_operator(self):
        I = []
        I.append(Observable(((self.raising_operator() + self.lowering_operator())/2).matrix))
        I.append(Observable(((self.raising_operator() - self.lowering_operator())/(2j)).matrix))
        I.append(Observable(self.d))
        for m in range(self.d):
            I[2].matrix[m, m] = self.quantum_number - m
        return I