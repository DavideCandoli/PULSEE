from Simulation import spin, Compute_Canonical_Density_Matrix, Trace
import numpy as np
import math
import hypothesis.strategies as st
from hypothesis import given
from hypothesis import assume

def test_Canonical_Density_Matrix_Unit_Trace():
    matrix = Compute_Canonical_Density_Matrix()
    trace = Trace(matrix)
    assert math.isclose(np.real(trace), 1, rel_tol=1e-10)
    
def test_Canonical_Density_Matrix_Hermitianity():
    matrix = Compute_Canonical_Density_Matrix()
    matrix_dagger = np.conj(matrix.transpose())
    assert np.all(np.isclose(matrix, matrix_dagger, rtol=1e-10))
    
@given(coefficients=st.lists(st.complex_numbers(max_magnitude = 128), min_size=spin.d, max_size=spin.d))
def test_Canonical_Density_Matrix_Positivity(coefficients):
    assume(np.all(np.isfinite(coefficients)))
    matrix = Compute_Canonical_Density_Matrix()
    vector = np.zeros(spin.d, dtype=complex)
    for m in range(spin.d):
        vector[m]=coefficients[m]
    ket = vector.T
    bra = np.conj(vector)
    positivity = np.real(bra@matrix@ket) >= 0
    assert positivity