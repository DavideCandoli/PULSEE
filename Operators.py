import math
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import expm, eig
from scipy.constants import hbar
from scipy.integrate import quad
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
            
    # The / operator acts between a Operator object (left) and a complex number (right) and returns the
    # operator divided by a this latter
    def __truediv__(self, divisor):
        try:
            divisor = complex(divisor)
            if divisor == 0:
                raise ZeroDivisionError                
            return Operator(self.matrix/divisor)
        except ValueError:
            raise ValueError("Invalid type for the right operand of /: the divisor must be a complex number")
        except ZeroDivisionError:
            raise ZeroDivisionError("The division of an Operator by 0 makes no sense")

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
    
    # Casts the Operator into the type of the subclass Density_Matrix (if all requirements are met)
    def cast_to_Density_Matrix(self):
        return Density_Matrix(self.matrix)
    
    # Casts the Operator into the type of the subclass Observable (if it is hermitian)
    def cast_to_Observable(self):
        return Observable(self.matrix)

    # Tries to cast the Operator into a Density_Matrix object, and if this step succeeds it performs the
    # evolution under the effect of a stationary Hamiltonian throughout a time interval 'time'
    def free_evolution(self, stat_hamiltonian, time):
        dm = self.cast_to_Density_Matrix()
        return dm.free_evolution(stat_hamiltonian, time)
    
    # Tries to cast the Operator into an Observable object, and if this step succeeds it computes the
    # expectation value of this Observable in the state represented by the given density matrix
    def expectation_value(self, density_matrix):
        ob = self.cast_to_Observable()
        return ob.expectation_value(density_matrix)


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

    # Makes the Density_Matrix evolve under the effect of a stationary Hamiltonian throughout a time
    # interval 'time'
    def free_evolution(self, stat_hamiltonian, time):
        iHt = (1j*stat_hamiltonian*float(time))
        evolved_dm = self.sim_trans(iHt, exp=True)
        return Density_Matrix(evolved_dm.matrix)


# Objects of the class Observable are hermitian operators representing the measurable properties of the
# system.
class Observable(Operator):
    
    # An instance of Observable is initialised in the same way as an Operator, the only difference being
    # that when a square array is passed to the constructor, this latter checks if it is hermitian and
    # raises error if it is not
    def __init__(self, x):
        ob_operator = Operator(x)
        if isinstance(x, np.ndarray):
            if not ob_operator.check_hermitianity():
                raise ValueError("The input array is not hermitian")
        self.matrix = ob_operator.matrix
        
    # Computes the expectation value of the observable in the state represented by 'density_matrix'
    # If the modulus of the imaginary part of the result is lower than 10^(-10), it returns a real
    # number
    def expectation_value(self, density_matrix):
        dm = density_matrix.cast_to_Density_Matrix()
        exp_val = (self*density_matrix).trace()
        if np.absolute(np.imag(exp_val)) < 1e-10: exp_val = np.real(exp_val)
        return exp_val


# Generates a random Operator whose elements are complex numbers with real and imaginary parts in the
# range [-10., 10.)
def Random_Operator(d):
    round_elements = np.vectorize(round)
    real_part = round_elements(20*(np.random.random_sample(size=(d, d))-1/2), 2)
    imaginary_part = 1j*round_elements(20*(np.random.random_sample(size=(d, d))-1/2), 2)
    random_array = real_part + imaginary_part
    return Operator(random_array)


# Generates a random Operator equal to its adjoint, whose elements are complex numbers with both real and imaginary parts in the range [-10., 10.)
def Random_Observable(d):
    o_random = Random_Operator(d)
    o_hermitian_random = (o_random + o_random.dagger())*(1/2)
    return Observable(o_hermitian_random.matrix)


# Generates a random Density_Matrix object, whose eigenvalues belong to the interval [0, 10.)
def Random_Density_Matrix(d):
    spectrum = 10*np.random.random(d)
    spectrum_norm = spectrum/(spectrum.sum())
    dm_diag = Density_Matrix(np.diag(spectrum_norm))
    cob = (1j*Random_Observable(d))  # The exponential of this matrix is a generic unitary transformation
    dm = dm_diag.sim_trans(cob, exp=True)
    return Density_Matrix(dm.matrix)


# Computes the commutator of two Operator objects
def Commutator(A, B):
    return A*B - B*A
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    