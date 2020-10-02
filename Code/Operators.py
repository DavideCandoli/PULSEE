import math
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import expm, eig
from scipy.constants import Planck, Boltzmann
from scipy.integrate import quad

# Objects of the class Operator represent linear applications which act on the vectors of a Hilbert space
class Operator:
    
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
                raise IndexError("An Operator object should be initialised with a 2D square array")
            cast_array_into_complex = np.vectorize(complex)
            input_array = cast_array_into_complex(x)
            self.matrix = input_array # Matrix representation of the operator (in the desired basis)
        else:
            d = int(x)
            self.matrix = np.identity(d, dtype=complex)

    def dimension(self):
        return self.matrix.shape[0]

    def __add__(self, addend):
        return Operator(self.matrix+addend.matrix)

    def __sub__(self, subtrahend):
        return Operator(self.matrix-subtrahend.matrix)
    
    def __neg__(self):
        return Operator(-self.matrix)
    
    def __mul__(self, factor):
        if isinstance(factor, Operator):
            return Operator(self.matrix@factor.matrix)
        else:
            factor = complex(factor)
            return Operator(self.matrix*factor)
                
    def __rmul__(self, factor):
        factor = complex(factor)
        return Operator(factor*self.matrix)
            
    def __truediv__(self, divisor):
        try:
            divisor = complex(divisor)
            if divisor == 0:
                raise ZeroDivisionError                
            return Operator(self.matrix/divisor)
        except ZeroDivisionError:
            raise ZeroDivisionError("The division of an Operator by 0 makes no sense")

    def __pow__(self, exponent):
        return Operator(np.linalg.matrix_power(self.matrix, exponent))
    
    def exp(self):
        exp_matrix = expm(self.matrix)
        return Operator(exp_matrix)
    
    def diagonalisation(self):
        eigenvalues, change_of_basis = np.linalg.eig(self.matrix)
        return eigenvalues, Operator(change_of_basis)
        
    # Performs a similarity transformation P^(-1)*M*P on the Operator M according to the given Operator
    # P for the change of basis
    def sim_trans(self, change_of_basis_operator, exp=False):
        if exp==True:
            left_exp = (-change_of_basis_operator).exp()
            right_exp = change_of_basis_operator.exp()
            new_basis_operator = left_exp*self*right_exp
        else:
            new_basis_operator = (change_of_basis_operator**(-1))*self*change_of_basis_operator
        return new_basis_operator

    def trace(self):
        trace = 0
        for i in range(self.dimension()):
            trace = trace + self.matrix[i, i]
        return trace

    def dagger(self):
        adjoint_matrix = (np.conj(self.matrix)).T
        return Operator(adjoint_matrix)

    # Returns the same Operator expressed in the quantum dynamical picture induced by the
    # Operator h_change_of_picture which is passed as an argument.
    # Passing a parameter `invert=True` yields the opposite transformation, bringing an Operator back
    # to the traditional Schroedinger picture
    def changed_picture(self, h_change_of_picture, time, invert=False):
        T = -1j*2*math.pi*h_change_of_picture*time
        if invert: T = -T
        return self.sim_trans(T, exp=True)

    def hermitianity(self):
        return np.all(np.isclose(self.matrix, self.dagger().matrix, rtol=1e-10))

    def unit_trace(self):
        return np.isclose(self.trace(), 1, rtol=1e-10)

    def positivity(self):
        eigenvalues = eig(self.matrix)[0]
        return np.all(np.real(eigenvalues) >= -1e-10)
    
    def cast_to_density_matrix(self):
        return Density_Matrix(self.matrix)
    
    def cast_to_observable(self):
        return Observable(self.matrix)

    def free_evolution(self, stat_hamiltonian, time):
        dm = self.cast_to_density_matrix()
        return dm.free_evolution(stat_hamiltonian, time)
    
    def expectation_value(self, density_matrix):
        ob = self.cast_to_observable()
        return ob.expectation_value(density_matrix)


# Objects of the class Density_Matrix are special Operator objects characterised by the following properties:
# i) Hermitianity;
# ii) Unit trace;
# iii) Positivity
class Density_Matrix(Operator):

    def __init__(self, x):
        d_m_operator = Operator(x)
        if isinstance(x, np.ndarray):
            error_message = "The input array lacks the following properties: \n"
            em = error_message
            if not d_m_operator.hermitianity():
                em = em + "- hermitianity \n"
            if not d_m_operator.unit_trace():
                em = em + "- unit trace \n"                
            if not d_m_operator.positivity():
                em = em + "- positivity \n"
            if em != error_message:
                raise ValueError(em)
        else:
            d = int(x)
            d_m_operator = d_m_operator*(1/d)
        self.matrix = d_m_operator.matrix

    def free_evolution(self, static_hamiltonian, time):
        iHt = (1j*2*math.pi*static_hamiltonian*time)
        evolved_dm = self.sim_trans(iHt, exp=True)
        return Density_Matrix(evolved_dm.matrix)


# Objects of the class Observable are hermitian operators representing the measurable properties of the
# system.
class Observable(Operator):
    
    def __init__(self, x):
        ob_operator = Operator(x)
        if isinstance(x, np.ndarray):
            if not ob_operator.hermitianity():
                raise ValueError("The input array is not hermitian")
        self.matrix = ob_operator.matrix
        
    def expectation_value(self, density_matrix):
        dm = density_matrix.cast_to_density_matrix()
        exp_val = (self*dm).trace()
        if np.absolute(np.imag(exp_val)) < 1e-10: exp_val = np.real(exp_val)
        return exp_val


def random_operator(d):
    round_elements = np.vectorize(round)
    real_part = round_elements(20*(np.random.random_sample(size=(d, d))-1/2), 2)
    imaginary_part = 1j*round_elements(20*(np.random.random_sample(size=(d, d))-1/2), 2)
    random_array = real_part + imaginary_part
    return Operator(random_array)


def random_observable(d):
    o_random = random_operator(d)
    o_hermitian_random = (o_random + o_random.dagger())*(1/2)
    return Observable(o_hermitian_random.matrix)


def random_density_matrix(d):
    spectrum = np.random.random(d)
    spectrum_norm = spectrum/(spectrum.sum())
    dm_diag = Density_Matrix(np.diag(spectrum_norm))
    cob = (1j*random_observable(d))  # The exponential of this matrix is a generic unitary transformation
    dm = dm_diag.sim_trans(cob, exp=True)
    return Density_Matrix(dm.matrix)


def commutator(A, B):
    return A*B - B*A


def magnus_expansion_1st_term(h, time_step):
    integral = h[0].matrix
    for t in range(len(h)-2):
        integral = integral + 2*h[t+1].matrix
    integral = (integral + h[-1].matrix)*(time_step)/2
    magnus_1st_term = Operator(-1j*2*math.pi*integral)
    return magnus_1st_term


def magnus_expansion_2nd_term(h, time_step):
    integral = (h[0]*0).matrix
    for t1 in range(len(h)-1):
        for t2 in range(t1+1):
            integral = integral + (commutator(h[t1], h[t2]).matrix)*(time_step**2)
    magnus_2nd_term = ((2*math.pi)**2)*Operator(-(1/2)*integral)
    return magnus_2nd_term


def magnus_expansion_3rd_term(h, time_step):
    integral = (h[0]*0).matrix
    for t1 in range(len(h)-1):
        for t2 in range(t1+1):
            for t3 in range(t2+1):
                integral = integral + \
                           ((commutator(h[t1], commutator(h[t2], h[t3])) + \
                             commutator(h[t3], commutator(h[t2], h[t1]))).matrix)*(time_step**3)
    magnus_3rd_term = Operator((1j/6)*((2*math.pi)**3)*integral)
    return magnus_3rd_term


def canonical_density_matrix(hamiltonian, temperature):
    
    if temperature <= 0:
        raise ValueError("The temperature must take a positive value")
    
    exponent = -(Planck*hamiltonian*1e6)/(Boltzmann*temperature)
    numerator = exponent.exp()
    canonical_partition_function = numerator.trace()
    canonical_dm = numerator/canonical_partition_function
    return Density_Matrix(canonical_dm.matrix)








