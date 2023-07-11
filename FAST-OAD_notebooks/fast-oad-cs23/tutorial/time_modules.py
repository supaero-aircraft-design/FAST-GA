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

    # Define relative path
    DATA_FOLDER_PATH = "data"
    WORK_FOLDER_PATH = "workdir"

    # Obtaining the modules for this version of fastoad
    command = "fastoad list_modules"
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Access the output using the 'stdout' attribute
    list_modules = result.stdout
    print(type(list_modules))

    module_times = dict.fromkeys(config_dictionary.keys(), 0)
    
    for module in config_dictionary.keys(): #iterate over each module present in the configuration file we are trying to optimize
        
        #generate the configuration file with single module
        CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "config_opti_tmp", str(module))

        shutil.copyfile(ORIGINAL_CONFIGURATION_FILE, CONFIGURATION_FILE)
        with open(CONFIGURATION_FILE, 'r') as file:
                data = yaml.safe_load(file)
                data['model']['aircraft_sizing'] = config_dictionary[module] 
        with open(CONFIGURATION_FILE, 'w') as file:
                if data['optimization']: del data['optimization'] 
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    
        ################################
        path_to_source = #parse list_modules to find the location of source files in: route-to-module just before  aerodynamics_low_speed.py file/unitary_tests/data/beechcraft_76.xml
        ###############################
        SOURCE_FILE = pth.join(DATA_FOLDER_PATH, "COMPLETEEEEEEE")

        api_cs25.generate_inputs(CONFIGURATION_FILE, SOURCE_FILE, overwrite=True)

        executions_time = []

        for _ in range(20): #run them individually 20 times, to have a good average

            starting = time.time()
            eval_problem = api_cs25.evaluate_problem(CONFIGURATION_FILE, overwrite=True)

            print('\n Problem ran in ', time.time() - starting , ' seconds \n')
            executions_time.append(time.time() - starting)

            """try:
                os.remove(WORK_FOLDER_PATH, "problem_outputs.xml")
            except:
                pass"""

        module_times[module] = sum(executions_time)/len(executions_time)

    

    return  module_times

    #remove all created yml files