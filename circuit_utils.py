import numpy as np
import Z_n_circuit_utils
from Circuits import unitary_sampler
import Circuits.circuit_evolution as evolution
from itertools import combinations
from scipy.special import binom


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
            p_list.append(np.array(p))
    
    indices = tuple(np.array(p_list).T)
    return indices

def get_indices_dic(L):
    return {q:get_indices(L,q) for q in range(L+1)}

def get_probability(state,L,Q,indices=[]):
    """
    This function return probability of having charge Q in the state
    Args:
        state ((2,)*L shape array): state of the quantum system
        L (_type_): system size
        Q (_type_): Charge

    Returns:
        prob: probability of having charge Q in the state
    """
    if not indices:
        indices = get_indices(L,Q)
    prob = np.sum(np.abs(state[indices])**2)
    return prob


def get_sharp_initial_state(L,Q):
    # Q = L//2 - N//2
    initial_state = np.zeros((2,)*L)

    for positions in combinations(range(L), Q):
        p = [0] * L
        for i in positions:
            p[i] = 1
        initial_state[tuple(p)] = 1 
    initial_state = initial_state/np.sum(np.abs(initial_state)**2)**0.5
    return initial_state


def get_fuzzy_initial_state(L,N):
    Q1 = L//2 - N//2
    Q2 = Q1 + N//2
    initial_state_1 = get_sharp_initial_state(L,Q1)
    initial_state_2 = get_sharp_initial_state(L,Q2)

    initial_state_1 = initial_state_1
    initial_state_2 = initial_state_2

    initial_state = (initial_state_1 + initial_state_2)/2**0.5

    assert round(np.sum(np.abs(initial_state)**2),10) == 1, str(np.sum(np.abs(initial_state)**2))+ ', ' + str(np.sum(initial_state_1!=0))
    
    return initial_state


def U_1_measurement(state,L,p_meas,m_locs=None,rng=None):
    if rng is None:
        seed = np.random.randint(1,1000000000)
        rng = np.random.default_rng(seed)
    if m_locs==None:
        m_locs = np.where(rng.uniform(0,1,L)<p_meas)[0]
    state,outcomes = evolution.measurement_layer(state,m_locs,rng)
    return state,outcomes

def U_1_layer(state,t,L,rng=None):
    if rng is None:
        seed = np.random.randint(1,1000000000)
        rng = np.random.default_rng(seed)
    U_list = [unitary_sampler.U_1_sym_gate_sampler(rng) for _ in range(L//2)]
    if t%2 == 0:
        state = evolution.evenlayer(state,U_list,L)
    else:
        state = evolution.oddlayer(state,U_list,L,BC='PBC')
    
    return state


def Z_n_measurement(state,L,N,Kraus_operators,g_N,m_locs=None,rng=None):
    if rng is None:
        seed = np.random.randint(1,1000000000)
        rng = np.random.default_rng(seed)
    if m_locs==None:
        m_locs = np.where(rng.uniform(0,1,L)<g_N)[0]
    
    outcomes = []
    for x in m_locs:
        locs = sorted(list(np.arange(x,x+N,1)%L))
        state,outcome = evolution.generalized_measurement(state,Kraus_operators,locs,rng)
        outcomes.append(outcome)
    
    return state, outcomes


def Z_n_layer(state,L,N,g_N,rng=None,gate_locs=None):
    if rng is None:
        seed = np.random.randint(1,1000000000)
        rng = np.random.default_rng(seed)

    if gate_locs is None:
        gate_locs = np.where(rng.uniform(0,1,L)<g_N)[0]

    for x in gate_locs:
        U_N = Z_n_circuit_utils.z_n_field(N,1)

        locs = sorted(list(np.arange(x,x+N,1)%L))
        state = evolution.generic_gate(state,U_N,location=locs)
    
    return state


def get_data_unitary_Z_N(L,N,T,p_meas,g_N,seed_U,seed_M,seed_N,initial_state=None,fuzzy_initial_state=True):
    if initial_state is None:
        if not fuzzy_initial_state:
            Q = L//2 - N//2
            state = get_sharp_initial_state(L,Q)
        else:
            state = get_fuzzy_initial_state(L,N)
    else:
        state = initial_state.copy()

    rng_N = np.random.default_rng(seed_N)
    rng_U = np.random.default_rng(seed_U)
    rng_M = np.random.default_rng(seed_M)

    indices_dic = get_indices_dic(L)
    prob_list = [[get_probability(state,L,q,indices=indices_dic[q]) for q in sorted(indices_dic.keys())]]

    # Krauss_operators = Z_n_circuit_utils.z_n_measurement(N)

    for t in range(T):
        
        state = Z_n_layer(state,L,N,g_N,rng=rng_N)
        state = U_1_layer(state,t,L,rng=rng_U)
        state,_ = U_1_measurement(state,L,p_meas,rng=rng_M)
        prob_list.append([get_probability(state,L,q,indices=indices_dic[q]) for q in sorted(indices_dic.keys())])

    return prob_list, state


def get_data_measurement_Z_N(L,N,T,p_meas,g_N,seed_U,seed_M,seed_N,initial_state=None,fuzzy_initial_state=True):
    if initial_state is None:
        if not fuzzy_initial_state:
            Q = L//2 - N//2
            state = get_sharp_initial_state(L,Q)
        else:
            state = get_fuzzy_initial_state(L,N)
    else:
        state = initial_state.copy()
    
    rng_N = np.random.default_rng(seed_N)
    rng_U = np.random.default_rng(seed_U)
    rng_M = np.random.default_rng(seed_M)

    indices_dic = get_indices_dic(L)
    prob_list = [[get_probability(state,L,q,indices=indices_dic[q]) for q in sorted(indices_dic.keys())]]

    Kraus_operators = Z_n_circuit_utils.z_n_measurement(N)

    for t in range(T):
        
        state,_ = Z_n_measurement(state,L,N,Kraus_operators,g_N,rng=rng_N)
        state = U_1_layer(state,t,L,rng=rng_U)
        state,_ = U_1_measurement(state,L,p_meas,rng=rng_M)
        prob_list.append([get_probability(state,L,q,indices=indices_dic[q]) for q in sorted(indices_dic.keys())])

    return prob_list, state


def get_data_phase_measurement_Z_N(L,N,T,p_meas,g_N,seed_U,seed_M,seed_N,initial_state=None,fuzzy_initial_state=True):
    if initial_state is None:
        if not fuzzy_initial_state:
            Q = L//2 - N//2
            state = get_sharp_initial_state(L,Q)
        else:
            state = get_fuzzy_initial_state(L,N)
    else:
        state = initial_state.copy()
    rng_N = np.random.default_rng(seed_N)
    rng_U = np.random.default_rng(seed_U)
    rng_M = np.random.default_rng(seed_M)

    indices_dic = get_indices_dic(L)
    prob_list = [[get_probability(state,L,q,indices=indices_dic[q]) for q in sorted(indices_dic.keys())]]

    Kraus_operators = Z_n_circuit_utils.z_n_measurement(N)

    for t in range(T):
        
        state,_ = Z_n_measurement(state,L,N,Kraus_operators,g_N,rng=rng_N)
        state = U_1_layer(state,t,L,rng=rng_U)
        state,_ = U_1_measurement(state,L,p_meas,rng=rng_M)
        prob_list.append([get_probability(state,L,q,indices=indices_dic[q]) for q in sorted(indices_dic.keys())])

    return prob_list, state


def get_entropy(probs):
    non_zero = np.array([p for p in probs if p!=0])
    return -np.sum(non_zero*np.log(non_zero))

def get_Z_N_entropy(probs,L,N):
    charges = np.arange(0,L+1,1)
    Z_N_charges = np.arange(0,N,1)
    prob_z_n_charge = [0]*len(Z_N_charges)
    for i,q in enumerate(charges):
        z_n_q = q%N
        prob_z_n_charge[z_n_q] += probs[i]
    return get_entropy(prob_z_n_charge)

def rolling_average(data,err,window):
    N = len(data)
    # print(data.shape,data[0:min(0+window,N-1)])
    new_data = [np.average(data[i:min(i+window,N-1)]) for i in range(N)]
    if window!=1:new_err =  [np.std(data[i:min(i+window,N-1)])/window**0.5 for i in range(N)]
    else: new_err = err.copy()
    return np.array(new_data), np.array(new_err)


def Z_n_max_value(z_n_q:list,L,N):
    prob_list = []
    for q in z_n_q:
        prob_list.append(0)
        for temp in range(q,L,N):
            prob_list[-1] += binom(L,temp)
    return get_entropy(prob_list)


def max_value(initial_charge,L,N):
    charges = [initial_charge]
    temp = initial_charge + N
    while temp < L+1:
        charges.append(temp)
        temp = temp + N
    temp = initial_charge - N
    while temp >= 0:
        charges.append(temp)
        temp = temp - N
    p_q = []
    for q in charges:
        p_q.append(binom(L,q))
    p_q = np.array(p_q)
    p_q = p_q / np.sum(p_q)
     

    return get_entropy(p_q), sorted(charges)