import math
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import expm, eig
from scipy.constants import Planck, Boltzmann
from scipy.integrate import quad
import Pade_Exp

# Objects of the class Operator represent linear applications which act on the vectors of a Hilbert space
class Operator:
    
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            try:
                if x.shape[0] != x.shape[1] or len(x.shape)>2:
                    raise IndexError  # If x is not a square array, IndexError is raised
            except IndexError:
                raise IndexError("An Operator object should be initialised with a 2D square array")
            cast_array_into_complex = np.vectorize(complex)
            input_array = cast_array_into_complex(x)
            self.matrix = input_array
        else:
            d = int(x)
            # Matrix representation of the operator (in the desired basis)
            self.matrix = np.identity(d, dtype=complex)

    # Returns the dimensionality of the Hilbert space where the Operator acts
    def dimension(self):
        return self.matrix.shape[0]

    # The definition of + and - operators is self-explanatory
    def __add__(self, addend):
        return Operator(self.matrix+addend.matrix)

    def __sub__(self, subtrahend):
        return Operator(self.matrix-subtrahend.matrix)
    
    # Returns the Operator changed by sign
    def __neg__(self):
        return Operator(-self.matrix)
    
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
    
    # Returns the eigenvalues and the Operator for the diagonalisation of the Operator object
    def diagonalise(self):
        eigenvalues, change_of_basis = np.linalg.eig(self.matrix)
        return eigenvalues, Operator(change_of_basis)
        
    # Performs a similarity transformation P^(-1)*M*P on the Operator M according to the given Operator
    # P for the change of basis
    def sim_trans(self, change_of_basis_operator, exp=False):
        try:
            if not isinstance(change_of_basis_operator, Operator):
                raise TypeError
            if exp==True:
                left_exp = (-change_of_basis_operator).exp()
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
    def change_picture(self, o_change_of_picture, time, invert=False):
        T = 2*math.pi*o_change_of_picture*(-1j*float(time))
        if invert: T = -T
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
    
    # Casts the Operator into the type of the subclass xix (if all requirements are met)
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
        iHt = (1j*2*math.pi*stat_hamiltonian*float(time))
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
    spectrum = np.random.random(d)
    spectrum_norm = spectrum/(spectrum.sum())
    dm_diag = Density_Matrix(np.diag(spectrum_norm))
    cob = (1j*Random_Observable(d))  # The exponential of this matrix is a generic unitary transformation
    dm = dm_diag.sim_trans(cob, exp=True)
    return Density_Matrix(dm.matrix)


# Computes the commutator of two Operator objects
def Commutator(A, B):
    return A*B - B*A


# Computes the 1st order term of the Magnus expansion of the passed time-dependent Hamiltonian
def Magnus_Expansion_1st_Term(h, time_step):
    integral = h[0].matrix
    for t in range(len(h)-2):
        integral = integral + 2*h[t+1].matrix
    integral = (integral + h[-1].matrix)*(time_step)/2
    magnus_1st_term = 2*math.pi*Operator(-1j*integral)
    return magnus_1st_term


# Computes the 2nd order term of the Magnus expansion of the passed time-dependent Hamiltonian
def Magnus_Expansion_2nd_Term(h, time_step):
    integral = (h[0]*0).matrix
    for t1 in range(len(h)-1):
        for t2 in range(t1+1):
            integral = integral + (Commutator(h[t1], h[t2]).matrix)*(time_step**2)
    magnus_2nd_term = ((2*math.pi)**2)*Operator(-(1/2)*integral)
    return magnus_2nd_term


# Computes the 3rd order term of the Magnus expansion of the passed time-dependent Hamiltonian
def Magnus_Expansion_3rd_Term(h, time_step):
    integral = (h[0]*0).matrix
    for t1 in range(len(h)-1):
        for t2 in range(t1+1):
            for t3 in range(t2+1):
                integral = integral + \
                           ((Commutator(h[t1], Commutator(h[t2], h[t3])) + \
                             Commutator(h[t3], Commutator(h[t2], h[t1]))).matrix)*(time_step**3)
    magnus_3rd_term = ((2*math.pi)**3)*Operator((1j/6)*integral)
    return magnus_3rd_term


# Returns the Density_Matrix associated with a canonically distributed ensemble of nuclear spins
def Canonical_Density_Matrix(hamiltonian, temperature):
    
    if temperature <= 0:
        raise ValueError("The temperature must take a non negative value")
    
    exponent = -(Planck*hamiltonian*1e6)/(Boltzmann*temperature)
    numerator = exponent.exp()
    canonical_partition_function = numerator.trace()
    canonical_dm = numerator/canonical_partition_function
    return Density_Matrix(canonical_dm.matrix)








