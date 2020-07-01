import math
import numpy as np

from Operators import Operator, Density_Matrix, Observable

def Tensor_Product_Operator(A, B):
    
    d_A = A.dimension()
    d_B = B.dimension()
    
    cast_array_into_complex = np.vectorize(complex)
    A_tensor_B = cast_array_into_complex(np.zeros([d_A*d_B, d_A*d_B]))
    
    for i in range(d_A):
        for j in range(d_A):
            for ii in range(d_B):
                for jj in range(d_B):
                    A_tensor_B[i*d_A+ii, j*d_A+jj] = A.matrix[i][j]*B.matrix[ii][jj]
    
    if isinstance(A, Density_Matrix) and isinstance(B, Density_Matrix):
        return Density_Matrix(A_tensor_B)
    
    elif isinstance(A, Observable) and isinstance(B, Observable):
        return Observable(A_tensor_B)
    
    else:
        return Operator(A_tensor_B)