from Simulation import spin, Unitary_Sandwich, Trace, Interaction_Picture, H_quadrupole, H_Zeeman, H1
import numpy as np
import math
import hypothesis.strategies as st
from hypothesis import given
from hypothesis import assume

def Generate_Random_Operator():
    return np.random.random_sample((spin.d, spin.d))*256 - 128

def Generate_Random_Density_Matrix():
    spectrum = np.random.random_sample(spin.d)*256 - 128
    normalised_spectrum = spectrum / np.sum(spectrum)
    random_matrix = np.zeros((spin.d, spin.d))
    for m in range(spin.d):
        random_matrix[m] = spectrum[m]
    random_exponent_operator = Generate_Random_Operator()
    return Unitary_Sandwich(random_matrix, random_exponent_operator)

def test_Unitary_Sandwich_Conserves_Trace():
    for n_trial in range(64):
        test_matrix = Generate_Random_Density_Matrix()
        random_operator = Generate_Random_Operator()
        transformed_test_matrix = Unitary_Sandwich(test_matrix, random_operator)
        assert math.isclose(Trace(transformed_test_matrix), Trace(test_matrix), rel_tol=1e-10)
        
def test_Interaction_Picture_Reversibility():
    for n_trial in range(64):
        test_matrix = Generate_Random_Density_Matrix()
        test_matrix_IP = Interaction_Picture(test_matrix, Generate_Random_Operator)
        test_matrix_restored = Interaction_Picture(test_matrix_IP, Generate_Random_Operator, invert=True)
        tautology = np.all(np.isclose(test_matrix, test_matrix_restored))
        assert tautology
        
def test_H_quadrupole_Hermitianity():
    h_q = H_quadrupole()
    h_q_dagger = np.conj(h_q.transpose())
    assert np.all(np.isclose(h_q, h_q_dagger, rtol=1e-10))
    
def test_H_Zeeman_Hermitianity():
    h_z = H_Zeeman()
    h_z_dagger = np.conj(h_z.transpose())
    assert np.all(np.isclose(h_z, h_z_dagger, rtol=1e-10))
    
@given(time=st.floats())
def test_H1_Hermitianity(time):
    h1 = H1(time)
    h1_dagger = np.conj(h1.transpose())
    assert np.all(np.isclose(h1, h1_dagger, rtol=1e-10))