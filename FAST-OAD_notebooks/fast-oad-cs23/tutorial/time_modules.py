def time_modules(config_dictionary, ORIGINAL_CONFIGURATION_FILE):

    import warnings
    warnings.filterwarnings(action="ignore")

    import os.path as pth
    import os
    import openmdao.api as om
    from fastoad import api as api_cs25
    from fastga.command import api as api_cs23
    import subprocess
    import time
    import yaml
    import shutil
    import re

    def find_id_value(dictionary):
        if 'id' in dictionary:
            return dictionary['id']
        else:
            for value in dictionary.values():
                if isinstance(value, dict):
                    id_value = find_id_value(value)
                    if id_value is not None:
                        return id_value
        return None

    # Define relative path
    WORK_FOLDER_PATH = "workdir"

    # Obtaining the modules for this version of fastoad
    command = "fastoad list_modules"
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Access the output using the 'stdout' attribute
    list_modules = result.stdout
    module_path_pairs = re.findall(r'\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|', list_modules)
    module_path_pairs = [(module.strip(), path.strip()) for module, path in module_path_pairs]


    module_times = dict.fromkeys(config_dictionary.keys(), 0)


    for module in config_dictionary.keys(): #iterate over each module present in the configuration file we are trying to optimize

        #generate the configuration file with single module (no optimization and no propeller)
        NEW_CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, 'config_opti_tmp', str(module))
        if not pth.isdir(NEW_CONFIGURATION_FILE):
            os.makedirs(NEW_CONFIGURATION_FILE)
        shutil.copy(ORIGINAL_CONFIGURATION_FILE, NEW_CONFIGURATION_FILE)
        NEW_CONFIGURATION_FILE = pth.join(NEW_CONFIGURATION_FILE, os.path.basename(ORIGINAL_CONFIGURATION_FILE))
        with open(NEW_CONFIGURATION_FILE, 'r') as file:
                data = yaml.safe_load(file)
                #Deletes keys inside aircraft sizing except the solvers, if present and the module we want to time
                keys_to_delete = [key for key in data['model']['aircraft_sizing'] if key not in ['linear_solver', 'nonlinear_solver', module]]
                for key in keys_to_delete:
                    del data['model']['aircraft_sizing'][key]
                #data['model']['aircraft_sizing'] = config_dictionary[module] #this is simpler but eliminates the specified solvers.
        with open(NEW_CONFIGURATION_FILE, 'w') as file:
                if 'optimization' in data: del data['optimization'] #no optimization
                if 'propeller' in data['model']: del data['model']['propeller'] #no propeller
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    
        #Fetch the appropriate SOURCE file from unitary_tests of each module, with the paths to the modules listed by the fastoad list_modules ran before
        ################################
        #Find the id or ids of the module in question
        id_of_module = find_id_value(config_dictionary[module])

        #Find the path to its folder
        path_to_id_of_module = next((pair[1] for pair in module_path_pairs if id_of_module in pair[0]), None)
        #Find the path to its unitary tests folder, which contains the source file
        index = path_to_id_of_module.index("/models")  # Find the index of "/models"
        next_folder_index = path_to_id_of_module.find("/", index + len("/models") + 1) # Find the index of the folder immediately after models
             
        if module != 'weight':
            NEW_SOURCE_FILE = path_to_id_of_module[:next_folder_index] + '/unitary_tests/data/beechcraft_76.xml' #add the path to the source file used for the unitary test of module being timed
   
            ###############################

            api_cs25.generate_inputs(NEW_CONFIGURATION_FILE, NEW_SOURCE_FILE, overwrite=True)

            executions_time = []

            print('\n Starting timings of: ', module)
            for _ in range(20): #run them individually 20 times, to have a good average

                starting = time.time()
                eval_problem = api_cs25.evaluate_problem(NEW_CONFIGURATION_FILE, overwrite=True)

                print('\n Problem ran in ', time.time() - starting , ' seconds \n')
                executions_time.append(time.time() - starting)

            module_times[module] = sum(executions_time)/len(executions_time)
            print('\n Average time of ', module, ': ', module_times[module], ' seconds')
        else:
            module_times['weight'] = 3
         
    #remove all temporary config files created for unitary timing
    shutil.rmtree(pth.join(WORK_FOLDER_PATH, 'config_opti_tmp'))
    return  module_times