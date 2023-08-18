import os.path as pth
import os
import sys

# import openmdao.api as om
# from fastoad import api as api_cs25
# from fastga.command import api as api_cs23
import yaml
import shutil
import time
from feedback_extractor import feedback_extractor
from time_modules import find_id_value

from fastoad.io.configuration import FASTOADProblemConfigurator
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.utils.om_warnings import issue_warning


def double_swap_algorithm(
    problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria
):
    conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
    problem = conf.get_problem()

    dict_with_solvers = config_dictionary.copy()
    config_dictionary.pop("nonlinear_solver", None)
    config_dictionary.pop("linear_solver", None)

    keys = config_dictionary.keys()
    best_score = feedback_extractor(
        problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria, WORK_FOLDER_PATH
    )  # Initial score of the dictionary
    keys_list = list(keys)
    best_order = keys_list.copy()  # Initial order of keys
    print("\n Starting DOUBLE swap")
    print("____________________________________\n")
    print("Starting order: ", keys_list)
    counter = 0

    # Load config file into dictionary:
    with open(CONFIGURATION_FILE, "r") as file:
        existing_data = yaml.safe_load(file)

    swap_position = 0
    if len(keys_list) < 2:
        sys.exit("\n Error: Less than two modules in aircraft_sizing. No sequencing to optimize.\n")
    while counter < len(keys_list):
        counter = counter + 1
        print("\nLoop nº: ", counter)
        improvement = False

        # Find a better position for the last node
        for i in range(len(keys_list) - 1):

            # Swap last node with the ith node (doube swap)
            keys_list[i], keys_list[-1 - swap_position] = (
                keys_list[-1 - swap_position],
                keys_list[i],
            )

            existing_data["model"]["aircraft_sizing"] = {
                key: config_dictionary[key]
                for key in keys_list
                if key in config_dictionary  # fill dict with new order
            }
            if is_valid_order(
                keys_list, config_dictionary
            ):  # check if order is valid according to restrictions set in function
                print("Trying order: ", keys_list)

                problem.setup()
                #CHANGE ORDER IN THE OPENMDAO PROBLEM WITHOUT TOUCHING CONFIG FILE
                ac_sizing=problem.model.aircraft_sizing
                ac_sizing.set_order(list(existing_data["model"]["aircraft_sizing"].keys()))
                print('FOR DEBUG: I made it through the reordering')
                problem.setup()
                problem.final_setup()
                # convert problem to dictionary, as input for feedback_extractor
                case_id = None
                model_data = _get_viewer_data(problem, case_id=case_id)

                # evaluate the score of the proposed order
                # print('FOR DEBUG IN DOUBLE SWAP, CONFIG DICTIONARY IS:, (check if fame order as trying order)', config_dictionary)
                score = feedback_extractor(
                    model_data,
                    existing_data["model"]["aircraft_sizing"],
                    CONFIGURATION_FILE,
                    score_criteria,
                    WORK_FOLDER_PATH,  #######################################################################################################################################################################################################################################
                )

                if score < best_score:
                    best_score = score
                    best_order = keys_list
                    improvement = True
                    swap_position = 0
                    print("Swap Kept")
                    print("Current order: ", best_order)
                    break

                else:
                    keys_list[i], keys_list[-1 - swap_position] = (
                        keys_list[-1 - swap_position],
                        keys_list[i],
                    )  # Revert the swap because it is worse
                    print("Swap reverted")
                    print("Current order: ", keys_list)
            else:
                # print("\n\n FOR DEBUG: BAD ORDER", keys_list)
                keys_list[i], keys_list[-1 - swap_position] = (
                    keys_list[-1 - swap_position],
                    keys_list[i],
                )  # Revert the swap because order would not have run

        if not improvement:
            swap_position = swap_position + 1

    keys_list = best_order  # Get the best order found so far
    existing_data["model"]["aircraft_sizing"] = {
        key: config_dictionary[key] for key in keys_list if key in config_dictionary
    }
    # add back solvers, if they were there
    if "nonlinear_solver" in dict_with_solvers.keys():
        existing_data["model"]["aircraft_sizing"]["nonlinear_solver"] = dict_with_solvers[
            "nonlinear_solver"
        ]
    if "linear_solver" in dict_with_solvers:
        existing_data["model"]["aircraft_sizing"]["linear_solver"] = dict_with_solvers[
            "linear_solver"
        ]
    with open(CONFIGURATION_FILE, "w") as file:
        yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)
        file.flush()

    return keys_list


def single_swap_algorithm(
    problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria
):

    conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
    problem = conf.get_problem()

    dict_with_solvers = config_dictionary.copy()
    config_dictionary.pop("nonlinear_solver", None)
    config_dictionary.pop("linear_solver", None)
    keys = config_dictionary.keys()
    best_score = feedback_extractor(
        problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria, WORK_FOLDER_PATH
    )  # Initial score of the dictionary
    keys_list = list(keys)
    best_order = keys_list.copy()  # Initial order of keys
    print("\n Starting SINGLE swap")
    print("____________________________________\n")
    print("Starting order: ", keys_list)
    counter = 0

    # Load config file into dictionary:
    with open(CONFIGURATION_FILE, "r") as file:
        existing_data = yaml.safe_load(file)

    if len(keys_list) < 2:
        sys.exit("\n Error: Less than two modules in aircraft_sizing. No sequencing to optimize.\n")
    while counter < len(keys_list):
        swap_position = 0
        print("Loop: nº", counter)
        improvement = False

        # Find a better position for the last node
        for i in range(len(keys_list) - 1):

            # Move last node to top position, displacing others (single swap)
            shifted_element = keys_list.pop(-1 - counter)
            keys_list.insert(swap_position, shifted_element)

            existing_data["model"]["aircraft_sizing"] = {
                key: config_dictionary[key]
                for key in keys_list
                if key in config_dictionary  # fill dict with new order
            }
            if is_valid_order(
                keys_list, config_dictionary
            ):  # check if order is valid according to restrictions set in function
                print("Trying order: ", keys_list)

                #CHANGE ORDER IN THE OPENMDAO PROBLEM WITHOUT TOUCHING CONFIG FILE
                ac_sizing=problem.model.aircraft_sizing
                ac_sizing.set_order(list(existing_data["model"]["aircraft_sizing"].keys()))
                print('FOR DEBUG: I made it through the reordering')
                problem.setup()
                problem.final_setup()       
                # convert problem to dictionary, as input for feedback_extractor
                case_id = None
                model_data = _get_viewer_data(problem, case_id=case_id)

                # evaluate the score of the proposed order
                score = feedback_extractor(
                    model_data,
                    existing_data["model"]["aircraft_sizing"],
                    CONFIGURATION_FILE,
                    score_criteria,
                    WORK_FOLDER_PATH,
                )

                if score < best_score:
                    best_score = score
                    best_order = keys_list
                    improvement = True
                    swap_position = 0
                    print("Swap Kept")
                    print("Current order: ", best_order, "\n")
                    break

                else:
                    shifted_element = keys_list.pop(
                        swap_position
                    )  # undo the change because the order is worst than the best proposed
                    keys_list.insert(len(keys_list) - counter, shifted_element)
                    swap_position = (
                        swap_position + 1
                    )  ###########################################################
                    print("Swap reverted")
                    print("Current order: ", keys_list, "\n")
            else:
                # print("\n\n FOR DEBUG: BAD ORDER", keys_list)
                shifted_element = keys_list.pop(
                    swap_position
                )  # undo the change because the order would not run (invalid order)
                keys_list.insert(len(keys_list) - counter, shifted_element)
                swap_position = swap_position + 1

        if not improvement:
            counter = counter + 1

    keys_list = best_order  # Get the best order found so far
    existing_data["model"]["aircraft_sizing"] = {
        key: config_dictionary[key] for key in keys_list if key in config_dictionary
    }
    # add back solvers, if they were there
    if "nonlinear_solver" in dict_with_solvers.keys():
        existing_data["model"]["aircraft_sizing"]["nonlinear_solver"] = dict_with_solvers[
            "nonlinear_solver"
        ]
    if "linear_solver" in dict_with_solvers:
        existing_data["model"]["aircraft_sizing"]["linear_solver"] = dict_with_solvers[
            "linear_solver"
        ]
    with open(CONFIGURATION_FILE, "w") as file:
        yaml.dump(existing_data, file, default_flow_style=False, sort_keys=False)
        file.flush()
    return keys_list


def hybrid_swap_algorithm(
    problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria
):

    print("\n HYBRID SWAP: First double, then single:")

    keys_list = double_swap_algorithm(
        problem_dictionary, config_dictionary, CONFIGURATION_FILE, score_criteria
    )

    # Setup the problem again with the updated order from the double swap
    conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
    problem = conf.get_problem()

    problem.setup()
    problem.final_setup()
    case_id = None
    model_data = _get_viewer_data(problem, case_id=case_id)
    with open(CONFIGURATION_FILE, "r") as file:
        yaml_data = yaml.safe_load(file)
        aircraft_sizing_data = yaml_data["model"]["aircraft_sizing"]

    keys_list = single_swap_algorithm(
        model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria
    )
    return keys_list


def is_valid_order(keys_list, dictionary):
    # Check the restrictions

    list_of_ids = []
    for key in keys_list:
        list_of_ids.append(find_id_value(dictionary[key]))

    id_indices = {
        id: list_of_ids.index(id) for id in list_of_ids
    }  # if id in list_of_ids else None for id in list_of_ids}

    def get_module_index(module_id):
        return id_indices[module_id] if module_id in id_indices else None

    def is_id_starts_with(prefix, id_indices):
        return any(id_index.startswith(prefix) for id_index in id_indices)

    # Check the restrictions
    # if (is_id_starts_with('fastga.handling_qualities.', id_indices) and
    #   (is_id_starts_with('fastga.geometry.', id_indices) or is_id_starts_with('fastga.aerodynamics.', id_indices))):
    #    handling_qualities_indices = [get_module_index(id) for id in id_indices if id.startswith('fastga.handling_qualities.')]
    #    geometry_aerodynamics_indices = [get_module_index(id) for id in id_indices if id.startswith('fastga.geometry.') or id.startswith('fastga.aerodynamics.')]
    #    if any(hq_index < geo_aero_index for hq_index in handling_qualities_indices for geo_aero_index in geometry_aerodynamics_indices):
    #        return False

    # if (get_module_index('fastga.loop.wing_position') is not None and
    #   get_module_index('fastga.weight.legacy') is not None):
    #    if get_module_index('fastga.loop.wing_position') < get_module_index('fastga.weight.legacy'):
    #        return False

    # Aero has to be computed before performance
    if is_id_starts_with("fastga.performances.", id_indices) and is_id_starts_with(
        "fastga.aerodynamics.", id_indices
    ):
        performances_indices = [
            get_module_index(id) for id in id_indices if id.startswith("fastga.performances.")
        ]
        aerodynamics_indices = [
            get_module_index(id) for id in id_indices if id.startswith("fastga.aerodynamics.")
        ]
        if any(
            performances_index < aerodynamics_index
            for performances_index in performances_indices
            for aerodynamics_index in aerodynamics_indices
            if performances_index is not None and aerodynamics_index is not None
        ):
            print("FOR DEBUG: Avoided aero > performance violation")
            return False

    # wing area has to be computed before geometry
    #    if is_id_starts_with("fastga.loop.wing_area.", id_indices) and is_id_starts_with(
    #        "fastga.geometry.", id_indices
    #    ):
    #        geometry_indices = [
    #            get_module_index(id) for id in id_indices if id.startswith("fastga.geometry.")
    #        ]
    #        wing_area_indices = [
    #            get_module_index(id) for id in id_indices if id.startswith("fastga.loop.wing_area.")
    #        ]
    #        if any(
    #            wing_area_index > geometry_index
    #            for wing_area_index in wing_area_indices
    #            for geometry_index in geometry_indices
    #            if wing_area_index is not None and geometry_index is not None
    #        ):
    #            print('FOR DEBUG: Avoided wing_area > geo violation')
    #            return False

    # Geometry has to be computed before aero
    if is_id_starts_with("fastga.geometry.", id_indices) and is_id_starts_with(
        "fastga.aerodynamics.", id_indices
    ):
        geometry_indices = [
            get_module_index(id) for id in id_indices if id.startswith("fastga.geometry.")
        ]
        aerodynamics_indices = [
            get_module_index(id) for id in id_indices if id.startswith("fastga.aerodynamics.")
        ]
        if any(
            geometry_index > aerodynamics_index
            for geometry_index in geometry_indices
            for aerodynamics_index in aerodynamics_indices
            if geometry_index is not None and aerodynamics_index is not None
        ):
            print("FOR DEBUG: Avoided geo > aero violation")
            return False

    # if (is_id_starts_with('fastga.geometry.', id_indices) and
    #   (get_module_index('fastga.loop.wing_area') is not None or get_module_index('fastga.loop.wing_position') is not None)):
    #    geometry_indices = [get_module_index(id) for id in id_indices if id.startswith('fastga.geometry.')]
    #    wing_area_wing_pos_indices = [get_module_index('fastga.loop.wing_area'), get_module_index('fastga.loop.wing_position')]
    #    if any(geometry_index > wing_area_wing_pos_index for geometry_index in geometry_indices for wing_area_wing_pos_index in wing_area_wing_pos_indices if geometry_index is not None and wing_area_wing_pos_index is not None):
    #        return False

    # if (
    #    get_module_index("fastga.weight.legacy") is not None
    #    and get_module_index("fastga.loop.mtow") is not None
    # ):
    #    if get_module_index("fastga.weight.legacy") > get_module_index("fastga.loop.mtow"):
    #        return False

    return True


####################################################################################################################################################################################
# End of the functions
####################################################################################################################################################################################


# Define relative path
WORK_FOLDER_PATH = "workdir"

# Define file
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process_test.yml")
start = time.time()

# Define optimizatin level: (assumes presence of aircraft_sizing, with modules defined)
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
swap = "single"  # Optimize using swap algorithm type: SINGLE or DOUBLE or HYBRID
# Optimize using as score:
#'use_time' pre-recorded single-module times multiplied by the times they run in feedbacks. Not all modules are present.
#'compute_time' live-recorded single-module times multiplied by the times they run in feedbacks - this will take longer as it has to run all your modules individually a few times
#'count_feedbacks' the count of how many feedback loops your config file has - quick and effective, for quick testing, or for general (but not thorough) optimization
score_criteria = "use_time"
############################################

if score_criteria == "compute_time":
    try:
        os.remove("tmp_saved_single_module_timings.txt")
    except FileNotFoundError:
        pass
try:
    # remove all temporary config files created for unitary timing
    shutil.rmtree(pth.join(WORK_FOLDER_PATH, "config_opti_tmp"))
except FileNotFoundError:
    pass

if optimization_level == 1:

    # Setup of the problem
    conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
    problem = conf.get_problem()
    problem.setup()
    problem.final_setup()

    # convert problem to dictionary
    case_id = None
    try:
        model_data = _get_viewer_data(problem, case_id=case_id)
        err_msg = ""
    except TypeError as err:
        model_data = {}
        err_msg = str(err)
        issue_warning(err_msg)

    with open(CONFIGURATION_FILE, "r") as file:

        # Load the YAML contents
        yaml_data = yaml.safe_load(file)
        # Get the structure under 'model: aircraft_sizing'
        try:
            yaml_data["model"]["aircraft_sizing"]
        except:
            sys.exit(
                "\n Error: Configuration optimizer assumes existence of aircraft_sizing inside model .yml config file \n"
            )
    aircraft_sizing_data = yaml_data["model"]["aircraft_sizing"]

    print(
        "\n Order before first swap: ",
        [
            key
            for key in aircraft_sizing_data
            if key != "linear_solver" and key != "nonlinear_solver"
        ],
    )

    if swap == "DOUBLE" or swap == "double":
        dummy_var = double_swap_algorithm(
            model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria
        )
    elif swap == "SINGLE" or swap == "single":
        dummy_var = single_swap_algorithm(
            model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria
        )
    elif swap == "HYBRID" or swap == "hybrid":
        dummy_var = hybrid_swap_algorithm(
            model_data, aircraft_sizing_data, CONFIGURATION_FILE, score_criteria
        )
    else:
        sys.exit("\n SWAP type not valid. Please choose SINGLE, DOUBLE or HYBRID \n")

# TODO::::
# elif Optimization_level == 2:
# elif Optimization_level == 3:
# elif Optimization_level == 4:
# elif Optimization_level == 5:
else:
    print("Not possible sry")

print(
    "\nYour configuration file has been overwritten with optimal order. Time taken",
    time.time() - start,
    "seconds",
)
# try:
#    os.remove('tmp_saved_single_module_timings.txt')
# except FileNotFoundError:
#    pass
try:
    # remove all temporary config files created for unitary timing
    shutil.rmtree(pth.join(WORK_FOLDER_PATH, "config_opti_tmp"))
except FileNotFoundError:
    pass
