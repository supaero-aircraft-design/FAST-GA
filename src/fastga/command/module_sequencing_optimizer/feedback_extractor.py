def feedback_extractor(
    model_data, score_criteria, WORK_FOLDER_PATH, orig_list_of_keys=None, CONFIGURATION_FILE=None, problem = None, INFO=False
):

    import time
    from time_modules import time_modules
    import sys

    # INFO: used to toggle on and off the printing of feedback loops and other info
    if INFO:
        start = time.time()
    import functools
    import os
    import json

    # definition of the function to output an n2-ordered list of all the variables in the problem
    # The dictionary is actually an alternation of lists of dicts where each dict has the key 'children', which contains a list of dicts, and so on
    def process_variables(children_list, prefix=""):
        variables = []
        for child in children_list:
            if "name" in child:
                var_name = prefix + child["name"]
                variables.append(var_name)
            if "children" in child:
                child_prefix = prefix + child["name"] + "."
                variables.extend(
                    process_variables(child["children"], child_prefix)
                )  # recursive call of the function to reach the last
        return variables

    def extract_blc(
        data,
    ):  # extracts the Bottom level components to which the feedbacking variables belong
        result = []
        for entry in data:
            src_value = entry["src"]
            tgt_value = entry["tgt"]

            src_parts = src_value.split(".")
            tgt_parts = tgt_value.split(".")

            src_word = src_parts[-2] if len(src_parts) >= 2 else ""
            tgt_word = tgt_parts[-2] if len(tgt_parts) >= 2 else ""

            result.append((src_word, tgt_word))

        return result

    def extract_module(data):
        result = []
        for entry in data:
            src_value = entry["src"]
            tgt_value = entry["tgt"]

            src_parts = src_value.split(".")
            tgt_parts = tgt_value.split(".")

            src_word = (
                src_parts[1] if len(src_parts) > 1 and src_parts[0] != "fastoad_shaper" else ""
            )
            tgt_word = (
                tgt_parts[1] if len(tgt_parts) > 1 and tgt_parts[0] != "fastoad_shaper" else ""
            )

            if src_word != "" and tgt_word != "":
                result.append((src_word, tgt_word))
        return result

    def total_time_of_modules(score_criteria, WORK_FOLDER_PATH):
        if score_criteria == "use_time":  # use pre-ran times for each individual module
            if os.path.exists("tmp_saved_single_module_timings.txt"):
                with open("tmp_saved_single_module_timings.txt", "r") as file:
                    module_times = json.loads(file.read())
                    return module_times
            else:
                print(
                    "\n Using pre-run module times. These may be different for your machine. It is recommended to run first with the option: compute_time"
                )
                modules_times = {
                    "geometry": 1.7216651797294618,
                    "aerodynamics_lowspeed": 2.5686846733093263,
                    "aerodynamics_highspeed": 2.165796732902527,
                    "weight": 3.7698839783668516,
                    "performance": 24.092622423171996,
                    "hq": 13.180185759067536,
                    "mtow": 1.0099695563316344,
                    "wing_position": 1.0157811045646667,
                    "wing_area": 1.1173424243927002,
                }
        else:  # compute modules times for your particular machine, solver, etc.
            modules_times = time_modules(orig_list_of_keys, CONFIGURATION_FILE, WORK_FOLDER_PATH)

        return modules_times

    def run_once(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists("tmp_saved_single_module_timings.txt"):
                result = func(*args, **kwargs)
                with open("tmp_saved_single_module_timings.txt", "w") as f:
                    f.write(json.dumps(result))
                return result
            else:
                with open("tmp_saved_single_module_timings.txt", "r") as f:
                    result = json.load(f)
                return result

        return wrapper

    # apply decorator
    time_modules = run_once(
        time_modules
    )  # Ensures that we only compute the time of the individual modules once

    #############################################
    # End of the functions
    #############################################

    # retrieves the dictionary for models and their sub-models and returns as an ordered list of the full "paths" to the variables
    ordered_vars = process_variables(model_data["tree"]["children"])

    # extracts connections for which the target is above the source, in the ordered list of variables
    connections_list = model_data["connections_list"]
    result_list = []
    distance_of_BLC = []
    for data in connections_list:
        src_position = ordered_vars.index(data["src"])
        tgt_position = ordered_vars.index(data["tgt"])
        if src_position > tgt_position:
            result_list.append(data)
            distance_of_BLC.append(
                src_position - tgt_position
            )  # register also how far apart they are - (at the time this is not used)

    # extract the BLCs from the variables
    list_of_blc_in_feedback = extract_blc(result_list)
    list_of_blc_in_feedback = [
        (a, b) for (a, b) in list_of_blc_in_feedback if "fastoad_shaper" not in (a, b)
    ]  # remove fastoad_shaper from feedback counts

    if score_criteria == "compute_time" or score_criteria == "use_time":

        modules_times = total_time_of_modules(score_criteria, WORK_FOLDER_PATH)

        # find how many times they run
        modules_in_feedback = extract_module(result_list)
        keys_order = orig_list_of_keys
        # print("FOR DEBUG: keys order is ", keys_order)

        rerun_counts = {
            key: 0 for key in keys_order
        }  # Initialize a dictionary to store rerun counts

        # count modules that have to be rerun
        for pair in modules_in_feedback:
            source, target = pair
            start_index = keys_order.index(source)  # Find the index of the source module
            end_index = keys_order.index(target)  # Find the index of the target module

            rerun_modules = keys_order[
                end_index : start_index + 1
            ]  # Extract the modules that need to be rerun

            for module in rerun_modules:
                rerun_counts[module] += 1  # Increment the rerun count (dict) for each module

        # Compute total time knowing how many reruns each module has:
        total_time = 0
        for module, count in rerun_counts.items():
            total_time += count * modules_times[module]
        score = total_time

    elif score_criteria == "count_feedbacks":
        score = len(list_of_blc_in_feedback)
    else:
        sys.exit(
            "\nScore criteria not valid. Please choose compute_time, use_time or count_feedbacks"
        )

    print("Score: ", score)

    if INFO:
        print("\n There are", len(list_of_blc_in_feedback), "feedback connections \n")

        # Prints the source and target of the feedback loop, along with the distance (in number of BLCs) that separates them
        print("Feedback loops:")
        loop_counter = 1
        loop_set = set()
        for pair in list_of_blc_in_feedback:
            loop_set.add(pair)
            print(
                f"{loop_counter}. {pair[0]} -> {pair[1]} | {distance_of_BLC[loop_counter-1]} BLCs"
            )
            loop_counter += 1
        print("\n")
        print("\n Time taken", time.time() - start, "seconds")

    return score
