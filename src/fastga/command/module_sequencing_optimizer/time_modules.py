import os.path as pth
import os
import openmdao.api as om
from fastoad import api as api_cs25
# from fastga.command import api as api_cs23
import time
import yaml
import shutil
import re
import sys
import contextlib
import xml.etree.ElementTree as ET
import copy

def find_id_value(dictionary):
    if "id" in dictionary:
        return dictionary["id"]
    else:
        for value in dictionary.values():
            if isinstance(value, dict):
                id_value = find_id_value(value)
                if id_value is not None:
                    return id_value
    return None

def time_modules(config_dictionary = None, ORIGINAL_CONFIGURATION_FILE = None, WORK_FOLDER_PATH = None, problem = None):

    from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

    NEW_SOURCE_FILE = pth.join("src/fastga/command/module_sequencing_optimizer/data/oad_process_outputs.xml")
    # initialize empty dict for storing module times
    
    
    if ORIGINAL_CONFIGURATION_FILE is not None:

        module_times = dict.fromkeys(config_dictionary, 0)
        

        print("\n   Calculating individual times for your modules... \n")
        for module in config_dictionary.keys():  # iterate over each module present in the configuration file we are trying to optimize

            # generate the configuration file with single module (no optimization and no propeller)
            NEW_CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "config_opti_tmp", str(module))
            if not pth.isdir(NEW_CONFIGURATION_FILE):
                os.makedirs(NEW_CONFIGURATION_FILE)
            shutil.copy(ORIGINAL_CONFIGURATION_FILE, NEW_CONFIGURATION_FILE)
            NEW_CONFIGURATION_FILE = pth.join(NEW_CONFIGURATION_FILE, os.path.basename(ORIGINAL_CONFIGURATION_FILE))
            with open(NEW_CONFIGURATION_FILE, "r") as file:
                data = yaml.safe_load(file)
                # Deletes keys inside aircraft sizing except the solvers, if present and the module we want to time
                keys_to_delete = [
                    key
                    for key in data["model"]["aircraft_sizing"]
                    if key not in ["linear_solver", "nonlinear_solver", module]
                ]
                for key in keys_to_delete:
                    del data["model"]["aircraft_sizing"][key]
                # data['model']['aircraft_sizing'] = config_dictionary[module] #this is simpler but eliminates the specified solvers.
            with open(NEW_CONFIGURATION_FILE, "w") as file:
                if "optimization" in data:
                    del data["optimization"]  # no optimization
                if "propeller" in data["model"]:
                    del data["model"]["propeller"]  # no propeller
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)            

            api_cs25.generate_inputs(NEW_CONFIGURATION_FILE, NEW_SOURCE_FILE, overwrite=True)

            executions_time = []

            print("\n   Starting timings of: ", module)
            for _ in range(15):  # run them individually 15 times, to have a good average

                starting = time.time()

                # Supressing terminal outputs of modules:

                # Save the original standard output and standard error
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                # Redirect standard output and standard error to the null device
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
                with contextlib.redirect_stdout(
                    None
                ):  # supresses outputs by terminal of each small case execution
                    api_cs25.evaluate_problem(NEW_CONFIGURATION_FILE, overwrite=True)
                # Restore the original standard output and standard error
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                executions_time.append(time.time() - starting)

            module_times[module] = sum(executions_time) / len(executions_time)
            print("\n       Average time of ", module, ": ", module_times[module], " seconds\n")


        try:
            # remove all temporary config files created for unitary timing
            shutil.rmtree(pth.join(WORK_FOLDER_PATH, "config_opti_tmp"))
        except FileNotFoundError:
            pass



    ###################### WARNING, THIS IS INCOMPLETE AND DOES NOT WORK YET (TIMING OF INDIVIDUAL MODULES WHEN RUNNING FROM A PROBLEM INSTEAD OF A CONFIG FILE)

    else: #if user selects option to run from openmdao problem and not config file
        
        model = copy.copy(problem.model)
        

        modules_in_problem = list(model.aircraft_sizing._proc_info.keys())
        module_times = dict.fromkeys(modules_in_problem, 0)

            

        for module in modules_in_problem:

            executions_time = []
            component_to_extract =  getattr(problem.model.aircraft_sizing, module)


            list_of_variables = list_inputs(component_to_extract)



            inputs = get_indep_var_comp(list_of_variables, "src/fastga/command/module_sequencing_optimizer/", "oad_process_outputs.xml")

            #print(inputs["data:geometry:propulsion:nacelle:y"])
            #print(inputs.get_val("data:geometry:propulsion:nacelle:y"))
            #print(inputs.list_outputs())
            """
            #problem_copy.model.add_subsystem("ivc", inputs, promotes = ["*"])
            #problem_copy.model.add_subsystem("component", component_to_extract, promotes=["*"])
            #problem_copy.model.add_subsystem("fastoad_shaper", fastoad_shaper)

            try:
                problem_copy.model.nonlinear_solver = nls
            except:
                pass
            try:
                problem_copy.model.linear_solver = ls
            except:
                pass



            """
            for i in range(15):  # run them individually 15 times, to have a good average
                starting = time.time()

                with contextlib.redirect_stdout(None):  # supresses outputs by terminal of each small case execution
                    problem_copy = run_system(component_to_extract, inputs) 

                executions_time.append(time.time() - starting)

            module_times[module] = sum(executions_time) / len(executions_time)
            print(module_times)      


    return module_times
