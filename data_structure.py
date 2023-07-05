import pickle 
import os
import pandas as pd

class structure():
    def __init__(self,root_direc='',sub_direc = []) -> None:
        self.root_directory = root_direc
        self.directories_name = sub_direc

    def add_sub_direc(self,sub_direc):
        self.sub_directories.append(sub_direc)

    def print_abstract_path(self):
        path = self.root_directory + '/'
        for sub_path in self.directories_name:
            if type(sub_path) is str:
                path = path + sub_path + '/'
            elif type(sub_path) is tuple:
                N = len(sub_path)
                for j,sub_sub_path in enumerate(sub_path):
                    path = path + sub_sub_path

                    if j != N-1:
                        path = path + '_'
                    else:
                        path = path + '/'
        print(path)

class data_file(structure):
    def __init__(self, root_direc='', sub_direc=[],values=[]) -> None:
        assert len(values) == len(sub_direc), "Number of values should be same as number sub-directories"
        super().__init__(root_direc, sub_direc)
        self.values = {}
        self.path = root_direc + '/'
        self.path = self._add_path(self.path,sub_direc,values,self.values)
        self.sub_directories = sub_direc
        

    def _add_path(self,path,sub_direc:list,values:list,value_dict: dict):
        path = path
        for i,sub_path in enumerate(sub_direc):
            value = values[i]
            assert type(sub_path) in [str,tuple], "sub_directories list should contain either str or tuple of str"

            if type(sub_path) is str:
                assert sub_path not in value_dict, "The sub-directory already exist in the structure"

                if value is None:
                    path = path + sub_path + '/'
                else:
                    path = path + sub_path + '=' + str(value) + '/'
                value_dict[sub_path] = value


            elif type(sub_path) is tuple:
                N = len(sub_path)
                for j,sub_sub_path in enumerate(sub_path):
                    assert sub_sub_path not in value_dict, "The sub-directory paramter already exist in the structure"

                    assert type(sub_sub_path) is str

                    sub_value = value[j]
                    if sub_value is None:
                        path = path + sub_sub_path
                    else:
                        path = path + sub_sub_path + '=' + str(sub_value)
                    value_dict[sub_sub_path] = sub_value
                    if j != N-1:
                        path = path + '_'
                    else:
                        path = path + '/'
        return path


    def update_path(self,sub_direc:list,values:list):
        self.path = self._add_path(self.path,sub_direc,values,self.values)
        self.sub_directories.append(sub_direc)

    

    def make_directories(self):
        if not os.path.isdir(self.path):
            try: os.makedirs(self.path)
            except:
                return

    def save_data(self,filename:str,data):
        direcs = filename.split('/')
        values = [None]*(len(direcs)-1)
        final_path = self.path
        final_path = self._add_path(final_path,direcs[:-1],values,value_dict={})

        if not os.path.isdir(final_path):
            os.makedirs(final_path)
        
        final_file = final_path + direcs[-1]
        with open(final_file,'wb') as f:
            pickle.dump(data,f)
    

    def load_data(self,filename:str):
        direcs = filename.split('/')
        values = [None]*(len(direcs)-1)
        final_path = self.path
        final_path = self._add_path(final_path,direcs[:-1],values,value_dict={})
        final_file = final_path + direcs[-1]
        assert os.path.isfile(final_file), "File " + final_file + " doesn't exist"
        with open(final_file,'rb') as f:
            data = pickle.load(f)
        
        return data

    
    # def write_summary(filedir:str,filename:str,comments:str)
    #     if os.path.isfile(filedir + '/summary.csv'):
    #         with open(filedir + '/summary.csv','rb') as f:
    #             summary = pickle.load(f)
    #     else:
    #         summary = pd.DataFrame()
    #         summary_dict = {'File name':[],'data_created':[],'last_updated':[]}


