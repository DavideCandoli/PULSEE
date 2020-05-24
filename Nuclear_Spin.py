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
    
    # The constructor of Nuclear_Spin objects receives as an argument only the spin quantum number s
    # and checks that this is a half-integer number as expected. Then, all other attributes are
    # initialised from the quantum number s.
    def __init__(self, s=1):
        try:
            s = float(s)
        except:
            raise TypeError("The given spin quantum number cannot be interpreted as a float")
        if not math.isclose(int(2*s), 2*s, rel_tol=1e-10):
            raise ValueError("The given spin quantum number is not a half-integer number")
        self.quantum_number = s
    def multiplicity(self):
        pass
    def raising_operator(self):
        pass
    def lowering_operator(self):
        pass
    def cartesian_operator(self):
        pass