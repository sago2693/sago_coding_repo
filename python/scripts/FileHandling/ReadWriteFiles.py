import pickle

def write_pickle_file(path, object_to_save, file_name,print_result=False):
    with open(path + file_name + '.data','wb') as filehandle:
        pickle.dump(object_to_save,filehandle)
    
    if print_result:
        print(f"The file {path}{file_name}.data was created successfully")

def read_pickle_file(path, file_name):
    with open(path + file_name + '.data', 'rb') as filehandle:
        loaded_file = pickle.load(filehandle)
    return loaded_file