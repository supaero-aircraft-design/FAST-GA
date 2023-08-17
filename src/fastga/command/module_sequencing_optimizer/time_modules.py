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


def time_modules(config_dictionary, ORIGINAL_CONFIGURATION_FILE, WORK_FOLDER_PATH):

    import os.path as pth
    import os
    import openmdao.api as om
    from fastoad import api as api_cs25

    # from fastga.command import api as api_cs23
    import subprocess
    import time
    import yaml
    import shutil
    import re
    import sys
    import contextlib
    import xml.etree.ElementTree as ET

    # Obtaining the modules for this version of fastoad
    command = "fastoad list_modules"
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Access the output using the 'stdout' attribute
    list_modules = result.stdout
    module_path_pairs = re.findall(r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", list_modules)
    module_path_pairs = [(module.strip(), path.strip()) for module, path in module_path_pairs]

    # initialize empty dict for storing module times
    module_times = dict.fromkeys(config_dictionary.keys(), 0)

    print("\n   Calculating individual times for your modules... \n")
    for (
        module
    ) in (
        config_dictionary.keys()
    ):  # iterate over each module present in the configuration file we are trying to optimize

        # generate the configuration file with single module (no optimization and no propeller)
        NEW_CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "config_opti_tmp", str(module))
        if not pth.isdir(NEW_CONFIGURATION_FILE):
            os.makedirs(NEW_CONFIGURATION_FILE)
        shutil.copy(ORIGINAL_CONFIGURATION_FILE, NEW_CONFIGURATION_FILE)
        FOLDER_WITH_CSVS_SRC = WORK_FOLDER_PATH + "/workdir"
        FOLDER_WITH_CSVS_TGT = NEW_CONFIGURATION_FILE + "/workdir"
        shutil.copytree(
            FOLDER_WITH_CSVS_SRC, FOLDER_WITH_CSVS_TGT, dirs_exist_ok=True
        )  # copy needed csv files to new config file path
        NEW_CONFIGURATION_FILE = pth.join(
            NEW_CONFIGURATION_FILE, os.path.basename(ORIGINAL_CONFIGURATION_FILE)
        )
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

        # Fetch the appropriate SOURCE file from unitary_tests of each module, with the paths to the modules listed by the fastoad list_modules ran before

        # Find the id or ids of the module in question
        id_of_module = find_id_value(config_dictionary[module])

        # Find the path to its folder
        path_to_id_of_module = next(
            (pair[1] for pair in module_path_pairs if id_of_module in pair[0]), None
        )
        # Find the path to its unitary tests folder, which contains the source file
        index = path_to_id_of_module.index("/models")  # Find the index of "/models"
        next_folder_index = path_to_id_of_module.find(
            "/", index + len("/models") + 1
        )  # Find the index of the folder immediately after models

        if id_of_module != "fastga.weight.legacy":

            NEW_SOURCE_FILE = (
                path_to_id_of_module[:next_folder_index] + "/unitary_tests/data/beechcraft_76.xml"
            )  # add the path to the source file used for the unitary test of module being timed
            if id_of_module == "fastga.loop.mtow":
                NEW_SOURCE_FILE = (
                    path_to_id_of_module[:next_folder_index]
                    + "/mass_breakdown/unitary_tests/data/beechcraft_76.xml"
                )  # add the path to the source file used for the unitary test of module being timed

            ###############################
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

        elif id_of_module == "fastga.weight.legacy":

            CG_SOURCE_FILE = (
                path_to_id_of_module[:next_folder_index]
                + "/cg/unitary_tests/data/beechcraft_76.xml"
            )  # add the path to the source file used for the unitary test of module being timed

            # we have to include a missing line to the source file in FAST-GA/src/fastga/models/weight/cg, so weight can run on its own
            tree = ET.parse(CG_SOURCE_FILE)
            root = tree.getroot()

            # Find the <geometry> element
            geometry_element = root.find("data/geometry")
            if geometry_element is not None:
                # Create missing element
                new_element = ET.Element("wing_configuration")
                new_element.set("is_input", "True")
                new_element.text = "1.0"

                # Append the new element to the <geometry> element
                geometry_element.append(new_element)
            else:
                sys.exit(
                    "<geometry> element not found in the XML structure. Could not compute weight times"
                )

            # Save the modified XML back to a file
            WEIGHT_PATH = pth.join(WORK_FOLDER_PATH, "config_opti_tmp", str(module), "data")
            if not pth.isdir(WEIGHT_PATH):
                os.makedirs(WEIGHT_PATH)
            WEIGHT_SOURCE_FILE = pth.join(WEIGHT_PATH, "beechcraft_76.xml")
            tree.write(WEIGHT_SOURCE_FILE)

            api_cs25.generate_inputs(NEW_CONFIGURATION_FILE, WEIGHT_SOURCE_FILE, overwrite=True)

            executions_time = []

            print("\n   Starting timings of: ", module)
            for _ in range(15):  # run them individually 15 times, to have a good average

                starting = time.time()

                with contextlib.redirect_stdout(
                    None
                ):  # supresses outputs by terminal of each small case execution
                    api_cs25.evaluate_problem(NEW_CONFIGURATION_FILE, overwrite=True)

                executions_time.append(time.time() - starting)

            module_times[module] = sum(executions_time) / len(executions_time)

            print("\n       Average time of ", module, ": ", module_times[module], " seconds\n")

    try:
        # remove all temporary config files created for unitary timing
        shutil.rmtree(pth.join(WORK_FOLDER_PATH, "config_opti_tmp"))
    except FileNotFoundError:
        pass
    return module_times
