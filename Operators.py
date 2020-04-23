import numpy as np

# Objects of the class Operator represent linear applications which act on the vectors of a Hilbert space
class Operator:
    
    # An instance may be initialised in two alternative ways:
    # 1) when an integer x is passed, the constructor generates an identity operator of dimensions x;
    # 2) when a square array is passed, this is assigned directly the 'matrix' attribute
    def __init__(self, x):
        if isinstance(x, int):
            # Matrix representation of the operator (in the desired basis)
            self.matrix = np.identity(x, dtype=complex)
        else:
            assert x.shape[0] == x.shape[1], "An Operator object should be initialised with a square array"
            cast_array_into_complex = np.vectorize(complex)
            input_array = cast_array_into_complex(x)
            self.matrix = input_array
            
    # Returns the dimensionality of the Hilbert space where the Operator acts
    def dimension(self):
        return self.matrix.shape[0]

# Objects of the class Density_Matrix are special Operator objects characterised by the following properties:
# i) Hermitianity;
# ii) Unit trace;
# iii) Positivity
class Density_Matrix(Operator):
    pass

# Objects of the class Observable are hermitian operators representing the measurable properties of the
# system.
class Observable(Operator):
    pass