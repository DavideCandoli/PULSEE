import math
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import expm, eig
from scipy.constants import hbar
import Pade_Exp

# Objects of the class Operator represent linear applications which act on the vectors of a Hilbert space
class Operator:
    
    # An instance may be initialised in two alternative ways:
    # 1) when an integer x is passed, the constructor generates an identity operator of dimensions x;
    # 2) when a square array is passed, this is assigned directly the 'matrix' attribute
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            try:
                if x.shape[0] != x.shape[1]:
                    raise IndexError  # If x is not a square array, IndexError is raised
                cast_array_into_complex = np.vectorize(complex)
                input_array = cast_array_into_complex(x)  # If x is an array of values which cannot be cast into complex, ValueError is raised
                self.matrix = input_array
            except IndexError:
                raise IndexError("An Operator object should be initialised with a 2D square array")
            except ValueError:
                raise ValueError("There are some elements in the array which cannot be interpreted as complex numbers")
        else:
            try:
                d = int(x)
                # Matrix representation of the operator (in the desired basis)
                self.matrix = np.identity(d, dtype=complex)
            except ValueError:
                raise ValueError("The value of the scalar argument cannot be interpreted as an integer")
            except TypeError:
                raise TypeError("The type of the argument is not valid. An Operator must be initialised either with an integer number or an array.")

    # Returns the dimensionality of the Hilbert space where the Operator acts
    def dimension(self):
        return self.matrix.shape[0]

    # The definition of + and - operators is self-explanatory
    def __add__(self, addend):
        return Operator(self.matrix+addend.matrix)
    def __sub__(self, subtrahend):
        return Operator(self.matrix-subtrahend.matrix)
    
    # The * operator is set up to handle both the product between Operator objects and the
    # multiplication by a scalar
    def __mul__(self, factor):
        if isinstance(factor, Operator):
            return Operator(self.matrix@factor.matrix)
        else:
            try:
                factor = complex(factor)
                return Operator(self.matrix*factor)
            except:
                raise ValueError("Invalid type for the right operand of *: the allowed ones are instances of the class Operator or numbers")
    def __rmul__(self, factor):
        try:
            factor = complex(factor)
            return Operator(factor*self.matrix)
        except:
            raise ValueError("Invalid type for the right operand of *: the allowed ones are instances of the class Operator or numbers")

    # The definition of the ** operator is self-explanatory
    def __pow__(self, exponent:int):
        return Operator(np.linalg.matrix_power(self.matrix, exponent))
    
    # Returns the exponential of the Operator
    def exp(self):
        exp_matrix = Pade_Exp.expm(self.matrix, 45)
        return Operator(exp_matrix)

    # Performs a similarity transformation P^(-1)*M*P on the Operator M according to the given Operator
    # P for the change of basis
    def sim_trans(self, change_of_basis_operator, exp=False):
        try:
            if not isinstance(change_of_basis_operator, Operator):
                raise TypeError
            if exp==True:
                left_exp = (change_of_basis_operator*(-1)).exp()
                right_exp = change_of_basis_operator.exp()
                new_basis_operator = left_exp*self*right_exp
            else:
                new_basis_operator = (change_of_basis_operator**(-1))*self*change_of_basis_operator
            return Operator(new_basis_operator.matrix)
        except TypeError:
            raise TypeError("Invalid type for the matrix of the change of basis: it should be an Operator object")
        except LinAlgError as e:
            if "Singular matrix" in e.args[0]:
                raise LinAlgError("The matrix for the change of basis must be invertible")

    # Computes the trace of the Operator
    def trace(self):
        trace = 0
        for i in range(self.dimension()):
            trace = trace + self.matrix[i][i]
        return trace

    # Returns the adjoint of the Operators
    def dagger(self):
        adjoint_matrix = (np.conj(self.matrix)).T
        return Operator(adjoint_matrix)

    # Returns the same Operator expressed in the interaction (or Dirac) picture induced by the
    # (stationary) unperturbed_hamiltonian which is passed as an argument.
    # Passing a parameter `invert=True` yields the opposite transformation, bringing an Operator in the
    # Dirac picture to the traditional Schroedinger one
    def interaction_picture(self, unperturbed_hamiltonian, time, invert=False):
        T = unperturbed_hamiltonian*(-1j*float(time))
        if invert: T = (-1)*T
        return self.sim_trans(T, exp=True)

    # Checks if the Operator is hermitian
    def check_hermitianity(self):
        return np.all(np.isclose(self.matrix, self.dagger().matrix, rtol=1e-10))

    # Checks if the Operator has unit trace
    def check_unit_trace(self):
        return np.isclose(self.trace(), 1, rtol=1e-10)

    # Checks if the Operator is positive
    def check_positivity(self):
        eigenvalues = eig(self.matrix)[0]
        return np.all(np.real(eigenvalues) >= -1e-10)


# Objects of the class Density_Matrix are special Operator objects characterised by the following properties:
# i) Hermitianity;
# ii) Unit trace;
# iii) Positivity
class Density_Matrix(Operator):

    # An instance of Density_Matrix is constructed in the same way as an Operator, with two differences:
    # 1) When x is a square array, the constructor checks the validity of the defining properties of a
    #    density matrix and raises error when any of them is not verified.
    # 2) When x is an integer, the matrix attribute is initialised by default with a maximally entangled
    #    density matrix, which means the identity matrix divided by its dimensions
    def __init__(self, x):
        d_m_operator = Operator(x)
        if isinstance(x, np.ndarray):
            error_message = "The input array lacks the following properties: \n"
            em = error_message
            if not d_m_operator.check_hermitianity():
                em = em + "- hermitianity \n"
            if not d_m_operator.check_unit_trace():
                em = em + "- unit trace \n"                
            if not d_m_operator.check_positivity():
                em = em + "- positivity \n"
            if em != error_message:
                raise ValueError(em)
        else:
            d = int(x)
            d_m_operator = d_m_operator*(1/d)
        self.matrix = d_m_operator.matrix
    
    # Makes the Density_Matrix evolve under the effect of a stationary Hamiltonian throughout a time interval 'time'
    def free_evolution(self, stat_hamiltonian, time):
        U = (1j*stat_hamiltonian*float(time))
        evolved_dm = self.sim_trans(U, exp=True)
        return evolved_dm
        
# Objects of the class Observable are hermitian operators representing the measurable properties of the
# system.
class Observable(Operator):
    pass

# Generates a random Operator whose elements are complex numbers with real and imaginary parts in the range [-10., 10.)
def Random_Operator(d):
    round_elements = np.vectorize(round)
    real_part = round_elements(20*(np.random.random_sample(size=(d, d))-1/2), 2)
    imaginary_part = 1j*round_elements(20*(np.random.random_sample(size=(d, d))-1/2), 2)
    random_array = real_part + imaginary_part
    return Operator(random_array)

# Generates a random Operator equal to its adjoint
def Random_Hermitian(d):
    o_random = Random_Operator(d)
    o_hermitian_random = (o_random + o_random.dagger())*(1/2)
    return o_hermitian_random

# Computes the commutator of two Operator objects
def Commutator(A, B):
    return A*B - B*A