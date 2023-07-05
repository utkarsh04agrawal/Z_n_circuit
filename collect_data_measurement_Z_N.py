from circuit_utils import get_data_measurement_Z_N, get_data_phase_measurement_Z_N, get_entropy, get_Z_N_entropy
import data_structure
import numpy as np
import time 

N=2
L_list = [6,8,10,12,14,16,18][:-1]
samps = 10
# p_meas = [0.05,0.1,0.2,0.3,0.6]
# g_N = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6]

p_meas = [0,0.01,0.03,0.05,0.075,0.1,0.2]
g_N = [0,0.01,0.05,0.1,0.2]
depth_ratio = 5
sub_direcs = [('L','T','N'),('p','g')]
phase_measurement = True
fuzzy_initial_state = True

if phase_measurement:
    collect_func = get_data_phase_measurement_Z_N
    root_directory = 'data/phase_measurement_Z_n/'
else:
    collect_func = get_data_measurement_Z_N
    root_directory = 'data/measurement_Z_n/'


if fuzzy_initial_state:
    root_directory =  root_directory +  'U_1_fuzzy_initial'
else:
    root_directory = root_directory + 'U_1_sharp_initial'

for p in p_meas:
    for g in g_N:
        for L in L_list:
            T = int(depth_ratio*L)
            params = [(L,T,N),(p,g)]
            param_file = data_structure.data_file(root_directory,sub_direcs,params)
            
            data_dic = {}
            entropy_data = []
            Z_n_entropy = []
            prob_data = []
            seeds = {}
            seeds['M'] = []
            seeds['U'] = []
            seeds['N'] = []
            start = time.time()
            for _ in range(samps):
                seed_M, seed_U, seed_N = np.random.randint(1,1000000000,3)
                seeds['M'].append(seed_M)
                seeds['U'].append(seed_U)
                seeds['N'].append(seed_N)
                prob_list,_ = collect_func(L,N,T,p,g,seed_U=seed_U,seed_M=seed_M,seed_N=seed_N)
                prob_data.append(prob_list)
                entropy_data.append([get_entropy(p_list) for p_list in prob_list])
                Z_n_entropy.append([get_Z_N_entropy(p_list,L,N) for p_list in prob_list])
            print(f'L={L}, p={p}, g={g}, time={time.time()-start}')

            param_file.make_directories()

            entropy_data = np.array(entropy_data)
            data_dic['data'] = entropy_data
            data_dic['Z_n_entropy'] = Z_n_entropy
            data_dic['prob_data'] = np.array(prob_data)
            data_dic['average'] = np.average(entropy_data,axis=0)
            data_dic['std'] = np.std(entropy_data,axis=0)
            data_dic['seeds'] = seeds

            filename = 'shots='+str(samps) + '_' + str(np.random.randint(1,100000000000))
            param_file.save_data(filename,data_dic)
            
