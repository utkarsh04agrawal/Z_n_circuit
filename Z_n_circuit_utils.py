"""This module does following things:
1. Generate Z_n symmetric gates
"""

import numpy as np
from Circuits import unitary_sampler
from itertools import combinations

def z_n_field(N: int, seed):
    """Z_N magentic field gate on N qubits

    Args:
        N (int): Z_N symmetry
        seed : seed for random number generator
    """

    U = np.eye(2**N,dtype=complex)
    U = np.reshape(U,(2,)*(2*N))
    rng = np.random.default_rng(seed=seed)
    U_4 = unitary_sampler.haar_qr(2,rng)
    U[(0,)*(2*N)] = U_4[0,0]
    U[(1,)*(2*N)] = U_4[1,1]
    U[(0,)*N + (1,)*N] = U_4[0,1]
    U[(1,)*N + (0,)*N] = U_4[1,0]

    return U.reshape((2**N,2**N))


def get_indices(L,Q):
    """
    Get indices of the configuration with total charge equal to Q
    """
    count = 0
    p_list = []
    for positions in combinations(range(L), Q):
            p = [0] * L

            for i in positions:
                p[i] = 1
            count += 1
            p_list.append(tuple(p))

    return p_list


def z_n_phase_measurement(N):
    """Returns Krauss operators for performing Z_N measurement, that corresponds to phase measurement of the phase in the Z_n sector

    Args:
        N (_type_): Z_N symmetry
    Returns:
        Krauss_operator: list of Krauss operators for the measurement
    """

    Krauss_operators = []
    M = np.zeros((2,)*(2*N))
    for q in range(1,N,1):
        
        indices = get_indices(N,q)
        
        for index in indices:
            M[index+index] = 1
    Krauss_operators.append(M.reshape((2**N,2**N)).copy())

    for q in [0,N]:
        M = np.zeros((2,)*(2*N))
        if q==0:
            indices = get_indices(N,q)
            indices2 = get_indices(N,q+N)
            for index in indices:
                M[index+index] = 1/2
                for index2 in indices2:
                    M[index2+index2] = 1/2
                    M[index+index2] = 1/2
                    M[index2+index] = 1/2
        
        elif q == N:
            indices = get_indices(N,q)
            indices2 = get_indices(N,q-N)
            for index in indices:
                M[index+index] = 1/2
                for index2 in indices2:
                    M[index2+index2] = 1/2
                    M[index+index2] = -1/2
                    M[index2+index] = -1/2
        
        Krauss_operators.append(M.reshape((2**N,2**N)).copy())
    
    return Krauss_operators

def z_n_measurement(N):
    """Returns Krauss operators for performing Z_N measurement, that corresponds to measurement of (-1)^{(Z_1 + Z_2 + ... + Z_N)/N}

    Args:
        N (_type_): Z_N symmetry
    Returns:
        Krauss_operator: list of Krauss operators for the measurement
    """

    Krauss_operators = []
    for q in range(0,N+1,1):
        M = np.zeros((2,)*(2*N))
        indices = get_indices(N,q)
        
        if q != 0:
            for index in indices:
                M[index+index] = 1
        
        if q==0:
            indices = get_indices(N,q)
            indices2 = get_indices(N,q+N)
            for index in indices:
                M[index+index] = 1/2
                for index2 in indices2:
                    M[index2+index2] = 1/2
                    M[index+index2] = 1/2
                    M[index2+index] = 1/2
        
        elif q == N:
            indices = get_indices(N,q)
            indices2 = get_indices(N,q-N)
            for index in indices:
                M[index+index] = 1/2
                for index2 in indices2:
                    M[index2+index2] = 1/2
                    M[index+index2] = -1/2
                    M[index2+index] = -1/2
        
        Krauss_operators.append(M.reshape((2**N,2**N)).copy())
    
    return Krauss_operators
        