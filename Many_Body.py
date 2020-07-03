import math
import numpy as np

from Operators import Operator, Density_Matrix, Observable

# Returns the tensor product of two operators
def Tensor_Product_Operator(A, B):
    
    d_A = A.dimension()
    d_B = B.dimension()
    
    cast_array_into_complex = np.vectorize(complex)
    A_tensor_B = cast_array_into_complex(np.zeros([d_A*d_B, d_A*d_B]))
    
    for i in range(d_A):
        for j in range(d_A):
            for ii in range(d_B):
                for jj in range(d_B):
                    A_tensor_B[i*d_B+ii, j*d_B+jj] = A.matrix[i, j]*B.matrix[ii, jj]
    
    if isinstance(A, Density_Matrix) and isinstance(B, Density_Matrix):
        return Density_Matrix(A_tensor_B)
    
    elif isinstance(A, Observable) and isinstance(B, Observable):
        return Observable(A_tensor_B)
    
    else:
        return Operator(A_tensor_B)
    
# Returns the partial trace of an operator
def Partial_Trace(operator, subspace_dimensions, index_position):
    m = operator.matrix
    
    d = subspace_dimensions
    
    i = index_position
    
    n = len(subspace_dimensions)
    
    d_downhill = int(np.prod(d[i+1:n+1]))
        
    d_block = d[i]*d_downhill
        
    d_uphill = int(np.prod(d[0:i]))
        
    partial_trace = np.empty((d_downhill, d_downhill*(d_uphill+1)), dtype=np.ndarray)
    for j in range(d_uphill):
        p_t_row = np.empty((d_downhill, d_downhill), dtype=np.ndarray)
        for k in range(d_uphill):
            block = m[j*d_block:(j+1)*d_block, k*d_block:(k+1)*d_block]
            p_t_block = np.zeros((d_downhill, d_downhill))
            for l in range(d[i]):
                p_t_block = p_t_block + block[l*d_downhill:(l+1)*d_downhill, \
                                              l*d_downhill:(l+1)*d_downhill]
            p_t_row = np.concatenate((p_t_row, p_t_block), axis=1)
        partial_trace = np.concatenate((partial_trace, p_t_row), axis=0)
        
    partial_trace = partial_trace[d_downhill:,d_downhill:]
    
    if isinstance(operator, Density_Matrix):
        return Density_Matrix(partial_trace)
    
    elif isinstance(operator, Observable):
        return Observable(partial_trace)
    
    else:
        return Operator(partial_trace)
    
    
