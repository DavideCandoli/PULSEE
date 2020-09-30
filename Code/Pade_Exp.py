from numpy import asarray, inf, dot, floor, identity
from math import log
import numpy as np

# Local imports
from scipy.linalg import norm
from numpy.linalg import solve

def expm(A, q=13):
    """Compute the matrix exponential using Pade approximation.
    Parameters
    ----------
    A : array, shape(M,M)
        Matrix to be exponentiated
    q : integer
        Order of the Pade approximation
    Returns
    -------
    expA : array, shape(M,M)
        Matrix exponential of A
    """
    A = asarray(A)

    # Scale A so that norm is < 1/2
    nA = norm(A,inf)
    if nA==0:
        return identity(A.shape[0], A.dtype.char)
    val = log(nA, 2)
    e = int(floor(val))
    j = max(0,e+1)
    A = A / 2.0**j

    # Pade Approximation for exp(A)
    X = A
    c = 1.0/2
    N = identity(A.shape[0]) + c*A
    D = identity(A.shape[0]) - c*A
    for k in range(2,q+1):
        c = c * (q-k+1) / (k*(2*q-k+1))
        X = dot(A,X)
        cX = c*X
        N = N + cX
        if not k % 2:
            D = D + cX;
        else:
            D = D - cX;
    F = solve(D,N)
    for k in range(1,j+1):
        F = dot(F,F)
    return F