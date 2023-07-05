import os
import numpy as np
import pickle
from datetime import datetime
from data_structure import data_file
import pandas as pd
from circuit_utils import get_Z_N_entropy
print(os.getcwd())

def get_L(name:str):
    L = 0
    index2 = name.index('_T')
    index1 = name.index('L=')
    L = int(name[index1+2:index2])
    return L

def get_p(name:str):
    p = 0
    index1 = name.index('p=')
    index2 = name.index('_g=')
    p = float(name[index1+2:index2])
    if p == 0:
        return int(p)
    return p

def get_g(name:str):
    g = 0
    index1 = name.index('g=')
    g = float(name[index1+2:])
    if g == 0:
        return int(g)
    return g

def get_T(name:str):
    T = 0
    index2 = name.index('_N')
    index1 = name.index('T=')
    T = int(name[index1+2:index2])
    return T

def get_N(name:str):
    N = 0
    index1 = name.index('N=')
    N = int(name[index1+2:])
    return N


def _initialze_completed_list_file(completed_list_file,rerun=False):
    if not os.path.isfile(completed_list_file):
        completed_list = {'File':[0],'date merged':['YYYY-MM-DD-Hr-M'],'batch':[0],'comments':[0]}
        completed_list = pd.DataFrame(completed_list)
        completed_list.to_csv(completed_list_file,index=True)
    else:
        completed_list = pd.read_csv(completed_list_file,index_col=0)
    if rerun:
        completed_list = {'File':[0],'date merged':['YYYY-MM-DD-Hr-M'],'batch':[0],'comments':[0]}
        completed_list = pd.DataFrame(completed_list)
        completed_list.to_csv(completed_list_file,index=True) 
    previous_batch = list(completed_list['batch'])[-1]
    current_batch = previous_batch + 1
    return completed_list, current_batch


def merge(file_structure:data_file,rerun=False):
    root_direc = file_structure.root_directory
    print(os.listdir(root_direc))       
    file_path = file_structure.path
    if not os.path.isdir(file_path):
        return False

    if rerun:
        if os.path.exists(file_path+'/merged_data/'):
            [os.remove(file_path+'/merged_data/'+f) for f in os.listdir(file_path+'/merged_data/')]

    T = file_structure.values['T']
    merged_data = {'U_1_entropy':None,'probs':None,'seeds':{},'Z_n_entropy':None}

    # completed_list contains name of files already merged
    completed_list_file = file_path + '/completed_list.csv'
    completed_list, current_batch = _initialze_completed_list_file(completed_list_file,rerun=rerun)
    print(completed_list)

    for individual_files in os.listdir(file_path):
        if individual_files in list(completed_list['File']):
            continue
        if 'shots=' not in individual_files:
            continue

        data = file_structure.load_data(individual_files)
        if merged_data['U_1_entropy'] is None:
            merged_data['U_1_entropy'] = data['data'].copy()
        else:
            merged_data['U_1_entropy'] = np.concatenate((data['data'],merged_data['U_1_entropy']),axis=0)

        if 'prob_data' in data:
            
            Z_n_data = np.empty(np.array(data['prob_data']).shape[:2])
            for s in range(Z_n_data.shape[0]):
                for t in range(Z_n_data.shape[1]):
                    Z_n_data[s,t] = get_Z_N_entropy(data['prob_data'][s,t],L,N)

            if merged_data['probs'] is None:
                merged_data['probs'] = data['prob_data'].copy()
                merged_data['Z_n_entropy'] = Z_n_data.copy()
            else:
                merged_data['probs'] = np.concatenate((data['prob_data'],merged_data['probs']),axis=0)
                merged_data['Z_n_entropy'] = np.concatenate((Z_n_data,merged_data['Z_n_entropy']),axis=0)

            
            
        
        for seed_name in ['M','U','N']:
            if seed_name not in merged_data['seeds']:
                merged_data['seeds'][seed_name] = []
            merged_data['seeds'][seed_name].extend(data['seeds'][seed_name])
        
        completed_list.loc[len(completed_list)] = [individual_files,datetime.today().strftime('%Y-%m-%d-%H:%M'),int(current_batch),'none']

    

    completed_list.to_csv(completed_list_file,index=True)    
    if merged_data['U_1_entropy'] is not None:
        N_samps = len(merged_data['U_1_entropy'])
        merged_filename = 'merged_data/data_'+datetime.today().strftime('%Y-%m-%d-%H:%M')+'_samples='+str(N_samps)+'_batch='+str(current_batch)
        file_structure.save_data(merged_filename,merged_data)


root_directories = ['data/phase_measurement_Z_n/U_1_fuzzy_initial',
                    'data/measurement_Z_n/U_1_sharp_initial',
                    'data/measurement_Z_n/U_1_fuzzy_initial',
                    'data/unitary_Z_n/U_1_sharp_initial',
                    'data/unitary_Z_n/U_1_fuzzy_initial'][-1:]

sub_direcs = [('L','T','N'),('p','g')]
rerun = False # whether to erase the previous data and merge again

for root_directory in root_directories:
    for L_file in os.listdir(root_directory):
        if L_file == '.DS_Store':
            continue
        L,T,N = get_L(L_file),get_T(L_file),get_N(L_file)
        for p_file in os.listdir(root_directory+'/'+L_file+'/'):
            p,g = get_p(p_file),get_g(p_file)
            params = [(L,T,N),(p,g)]
            print(params)
            param_file = data_file(root_directory,sub_direcs,params)
            merge(param_file,rerun)

