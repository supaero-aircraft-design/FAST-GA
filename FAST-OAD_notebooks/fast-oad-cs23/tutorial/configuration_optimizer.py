import os.path as pth
import os
import sys
import openmdao.api as om
from fastoad import api as api_cs25
from fastga.command import api as api_cs23
import yaml
import shutil
import time
from feedback_extractor import *

from fastoad.io.configuration import FASTOADProblemConfigurator
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.utils.om_warnings import issue_warning

def double_swap_algorithm(problem_dictionary, config_dictionary, CONFIGURATION_FILE):

    keys = config_dictionary.keys()
    best_score = feedback_extractor(problem_dictionary)  # Initial score of the dictionary
    keys_list = list(keys)
    best_order = keys_list.copy()  # Initial order of keys

    print('Starting order: ', keys_list)
    counter = 0
    while True:
        counter =+ 1 
        print(counter)
        improvement = False
        # Find a better position for the last node
        for i in range(len(keys_list) - 1):

            # Swap last node with the ith node (doube swap)
            keys_list[i], keys_list[-1] = keys_list[-1], keys_list[i]  
            print('Trying order: ', keys_list)
            #Generate new config file with proposed order
            #new_ordered_config = {key: config_dictionary[key] for key in keys if key in config_dictionary}
            with open(CONFIGURATION_FILE, 'r') as file:
                existing_data = yaml.safe_load(file)


            #for key in keys_list:
            #    existing_data['model']['aircraft_sizing'][key] = config_dictionary[key]

            #existing_data['model']['aircraft_sizing'].update(keys) 
            #   
            existing_data['model']['aircraft_sizing'] = {key: config_dictionary[key] for key in keys_list if key in config_dictionary}

            with open(CONFIGURATION_FILE, 'w') as file:
                yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)

            
            conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
            problem = conf.get_problem()
            problem.setup()
            problem.final_setup()
            #convert problem to dictionary, as input for feedback_extractor
            case_id=None
            model_data = _get_viewer_data(problem, case_id=case_id)

            #evaluate the score of the proposed order
            score = feedback_extractor(model_data)
            
            if score < best_score:
                best_score = score
                best_order = list(keys).copy()
                improvement = True
                print('Swap Kept')
                print('Current order: ', best_order)

            else:
                keys_list[i], keys_list[-1] = keys_list[-1], keys_list[i]  # Revert the swap
                print('Swap reverted')
                print('Current order: ', keys_list)

        if not improvement:
            break
        
    keys_list = best_order.copy()  # Start from the best order found so far
    best_config = existing_data
    best_config['model']['aircraft_sizing'] = {key: config_dictionary[key] for key in keys_list if key in config_dictionary}
    #Write best configuration
    with open(CONFIGURATION_FILE, 'w') as file:
        yaml.dump(best_config, file, default_flow_style=False, sort_keys=False)
    print('Best order has been written in configuration file')
    return keys_list





# Define relative path
WORK_FOLDER_PATH = "workdir"

# Define file
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process_test.yml")


#Define optimizatin level: (assumes presence of aircraft_sizing, with modules defined)
# LVL 1: CONFIG FILE MODULES - shuffles the order of
#        geometry
#        aerodynamics_lowspeed
#        aerodynamics_highspeed
#        weight
#        performance
#        hq
#        mtow
#        wing_position
#        wing_area
#
# LVL 2: Shuffles order of all sub-modules in all the modules of the CONFIG file
# LVL 3: etc 
# LVL 4: etc
start = time.time()

optimization_level = 1

if optimization_level == 1:

    #Setup of the problem
    conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
    problem = conf.get_problem()
    problem.setup()
    problem.final_setup()

    #convert problem to dictionary
    case_id=None
    try:
        model_data = _get_viewer_data(problem, case_id=case_id)
        err_msg = ''
    except TypeError as err:
        model_data = {}
        err_msg = str(err)
        issue_warning(err_msg)


 
    with open(CONFIGURATION_FILE, 'r') as file:

        # Load the YAML contents
        yaml_data = yaml.safe_load(file)

        # Get the structure under 'model: aircraft_sizing'
        try:
            yaml_data['model']['aircraft_sizing']
        except:
            sys.exit("\n Error: Configuration optimizer assumes existence of aircraft_sizing inside model .yml config file \n")

        yaml_data['model']['aircraft_sizing'].pop('nonlinear_solver', None)
        yaml_data['model']['aircraft_sizing'].pop('linear_solver', None)
        aircraft_sizing_data = yaml_data['model']['aircraft_sizing']

        print('Starting order before first swap: ', aircraft_sizing_data.keys())
        dummy_var = double_swap_algorithm(model_data, aircraft_sizing_data, CONFIGURATION_FILE)

        print(dummy_var)


        #print('\n', feedback_extractor(aircraft_sizing_data))











#elif Optimization_level == 2:
#elif Optimization_level == 3:
#elif Optimization_level == 4:
#elif Optimization_level == 5:
else:
    print('Not possible sry')

print('\n Time taken', time.time() - start, 'seconds')