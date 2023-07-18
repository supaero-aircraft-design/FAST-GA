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

def double_swap_algorithm(problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria):
    
    dict_with_solvers = config_dictionary.copy()

    config_dictionary.pop('nonlinear_solver', None)
    config_dictionary.pop('linear_solver', None)

    keys = config_dictionary.keys()
    best_score = feedback_extractor(problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria, WORK_FOLDER_PATH)  # Initial score of the dictionary
    keys_list = list(keys)
    best_order = keys_list.copy()  # Initial order of keys
    print("\n Starting DOUBLE swap")
    print("____________________________________\n")
    print('Starting order: ', keys_list)
    counter = 0
    swap_position = 0
    if len(keys_list)<2: sys.exit("\n Error: Less than two modules in aircraft_sizing. No sequencing to optimize.\n")
    while counter < len(keys_list):
        counter = counter + 1 
        print('\nLoop nº: ', counter)
        improvement = False
        # Find a better position for the last node
        for i in range(len(keys_list) - 1):

            # Swap last node with the ith node (doube swap)
            keys_list[i], keys_list[-1-swap_position] = keys_list[-1-swap_position], keys_list[i]  
            
            print('Trying order: ', keys_list)
            #Generate new config file with proposed order
            with open(CONFIGURATION_FILE, 'r') as file:
                existing_data = yaml.safe_load(file) 
            existing_data['model']['aircraft_sizing'] = {key: config_dictionary[key] for key in keys_list if key in config_dictionary}
            with open(CONFIGURATION_FILE, 'w') as file:
                yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)
            #convert config file to problem and then to dictionary, as input for feedback_extractor
            conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
            problem = conf.get_problem()
            problem.setup()
            problem.final_setup()
            #convert problem to dictionary, as input for feedback_extractor
            case_id=None
            model_data = _get_viewer_data(problem, case_id=case_id)

            #evaluate the score of the proposed order
            score = feedback_extractor(model_data, config_dictionary, CONFIGURATION_FILE, score_criteria, WORK_FOLDER_PATH)
            
            if score < best_score:
                best_score = score
                best_order = keys_list
                improvement = True
                swap_position = 0
                print('Swap Kept')
                print('Current order: ', best_order)
                break

            else:
                keys_list[i], keys_list[-1-swap_position] = keys_list[-1-swap_position], keys_list[i]  # Revert the swap
                print('Swap reverted')
                print('Current order: ', keys_list)

        if not improvement: swap_position = swap_position + 1
        
    keys_list = best_order  # Start from the best order found so far
    existing_data['model']['aircraft_sizing'] = {key: config_dictionary[key] for key in keys_list if key in config_dictionary}
    #add back solvers, if they were there
    if 'nonlinear_solver' in dict_with_solvers.keys(): existing_data['model']['aircraft_sizing']['nonlinear_solver'] = dict_with_solvers['nonlinear_solver'] 
    if 'linear_solver' in dict_with_solvers: existing_data['model']['aircraft_sizing']['linear_solver'] = dict_with_solvers['linear_solver'] 
    with open(CONFIGURATION_FILE, 'w') as file:
        yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)
        file.flush()
    return keys_list


def single_swap_algorithm(problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria):
    
    dict_with_solvers = config_dictionary.copy()
    config_dictionary.pop('nonlinear_solver', None)
    config_dictionary.pop('linear_solver', None)
    keys = config_dictionary.keys()
    best_score = feedback_extractor(problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria, WORK_FOLDER_PATH)  # Initial score of the dictionary
    keys_list = list(keys)
    best_order = keys_list.copy()  # Initial order of keys
    print("\n Starting SINGLE swap")
    print("____________________________________\n")
    print('Starting order: ', keys_list)
    counter = 0
    swap_position = 0
    if len(keys_list)<2: sys.exit("\n Error: Less than two modules in aircraft_sizing. No sequencing to optimize.\n")
    while counter < len(keys_list):
        counter = counter + 1 
        print('Loop: nº', counter)
        improvement = False
        # Find a better position for the last node
        for i in range(len(keys_list) - 1):

            # Move last node to top position, displacing others
            shifted_element = keys_list.pop(-1-swap_position)
            keys_list.insert(0, shifted_element)
            print('Trying order: ', keys_list)
            #Generate new config file with proposed order
            with open(CONFIGURATION_FILE, 'r') as file:
                existing_data = yaml.safe_load(file) 
            existing_data['model']['aircraft_sizing'] = {key: config_dictionary[key] for key in keys_list if key in config_dictionary}
            with open(CONFIGURATION_FILE, 'w') as file:
                yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)

            #convert config file to problem and then to dictionary, as input for feedback_extractor
            conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
            problem = conf.get_problem()
            problem.setup()
            problem.final_setup()
            #convert problem to dictionary, as input for feedback_extractor
            case_id=None
            model_data = _get_viewer_data(problem, case_id=case_id)

            #evaluate the score of the proposed order
            score = feedback_extractor(model_data, config_dictionary, CONFIGURATION_FILE, score_criteria, WORK_FOLDER_PATH)
            
            if score < best_score:
                best_score = score
                best_order = keys_list
                improvement = True
                swap_position = 0
                print('Swap Kept')
                print('Current order: ', best_order)
                break

            else:
                shifted_element = keys_list.pop(0)
                keys_list.insert(-1-swap_position, shifted_element)
                print('Swap reverted')
                print('Current order: ', keys_list)

        if not improvement:
            swap_position = swap_position + 1
        

    keys_list = best_order
    existing_data['model']['aircraft_sizing'] = {key: config_dictionary[key] for key in keys_list if key in config_dictionary}
    #add back solvers, if they were there
    if 'nonlinear_solver' in dict_with_solvers.keys(): existing_data['model']['aircraft_sizing']['nonlinear_solver'] = dict_with_solvers['nonlinear_solver'] 
    if 'linear_solver' in dict_with_solvers: existing_data['model']['aircraft_sizing']['linear_solver'] = dict_with_solvers['linear_solver'] 
    with open(CONFIGURATION_FILE, 'w') as file:
        yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)
        file.flush()
    return keys_list


def hybrid_swap_algorithm(problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria):

    print("\n HYBRID SWAP: First double, then single:")

    keys_list = double_swap_algorithm(problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria)

    #Setup the problem again with the updated order from the double swap
    conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
    problem = conf.get_problem()
    problem.setup()
    problem.final_setup()
    case_id=None
    model_data = _get_viewer_data(problem, case_id=case_id)
    with open(CONFIGURATION_FILE, 'r') as file:
        yaml_data = yaml.safe_load(file)
        aircraft_sizing_data = yaml_data['model']['aircraft_sizing']

    keys_list = single_swap_algorithm(model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria)
    return keys_list



####################################################################################################################################################################################
#End of the functions
####################################################################################################################################################################################


# Define relative path
WORK_FOLDER_PATH = "workdir"

# Define file
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process_test.yml")
start = time.time()

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
#        etc
# LVL 2: Shuffles order of all sub-modules in all the modules of the CONFIG file TO BE DONE
# LVL 3: etc TO BE DONE
# LVL 4: etc TO BE DONE

############################################
optimization_level = 1
swap = 'single' #Optimize using swap algorithm type: SINGLE or DOUBLE or HYBRID
#Optimize using as score: 
    #'use_time' pre-recorded single-module times multiplied by the times they run in feedbacks. Not all modules are present.  
    #'compute_time' live-recorded single-module times multiplied by the times they run in feedbacks - this will take longer as it has to run all your modules individually a few times
    #'count_feedbacks' the count of how many feedback loops your config file has - quick and effective, for quick testing, or for general (but not thorough) optimization
score_criteria = 'compute_time'  
############################################

try:
    os.remove('tmp_saved_single_module_timings.txt')
except FileNotFoundError:
    pass
try:
    #remove all temporary config files created for unitary timing
    shutil.rmtree(pth.join(WORK_FOLDER_PATH, 'config_opti_tmp'))
except FileNotFoundError:
    pass
    
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
    aircraft_sizing_data = yaml_data['model']['aircraft_sizing']

    print('\n Order before first swap: ', [key for key in aircraft_sizing_data if key != 'linear_solver' and key != 'nonlinear_solver'])

    if swap == 'DOUBLE' or swap == 'double':
        dummy_var = double_swap_algorithm(model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria)
    elif swap == 'SINGLE' or swap == 'single':
        dummy_var = single_swap_algorithm(model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria)
    elif swap == 'HYBRID' or swap == 'hybrid':
        dummy_var = hybrid_swap_algorithm(model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria)
    else: sys.exit("\n SWAP type not valid. Please choose SINGLE, DOUBLE or HYBRID \n")

#TODO::::
#elif Optimization_level == 2:
#elif Optimization_level == 3:
#elif Optimization_level == 4:
#elif Optimization_level == 5:
else:
    print('Not possible sry') 

print('\nYour configuration file has been overwritten with optimal order. Time taken', time.time() - start, 'seconds')
try:
    os.remove('tmp_saved_single_module_timings.txt')
except FileNotFoundError:
    pass
try:
    #remove all temporary config files created for unitary timing
    shutil.rmtree(pth.join(WORK_FOLDER_PATH, 'config_opti_tmp'))
except FileNotFoundError:
    pass