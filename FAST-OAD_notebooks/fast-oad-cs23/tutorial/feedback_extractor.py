def feedback_extractor(model_data, INFO = False):

    import time

    #INFO: used to toggle on and off the printing of feedback loops and other info
    if INFO: start = time.time()

    #definition of the function to output an n2-ordered list of all the variables in the problem
    #The dictionary is actually an alternation of lists of dicts where eahc dicht has the key 'children', which contains a list of dicts, and so on
    def process_variables(children_list, prefix=''):
        variables = []
        for child in children_list:
            if 'name' in child:
                var_name = prefix + child['name']
                variables.append(var_name)
            if 'children' in child:
                child_prefix = prefix + child['name'] + '.'
                variables.extend(process_variables(child['children'], child_prefix)) #recursive call of the function to reach the last 
        return variables


    def extract_BLC(data): #extracts the Bottom level components to which the feedbacking variables belong
        result = []
        for entry in data:
            src_value = entry['src']
            tgt_value = entry['tgt']

            src_parts = src_value.split('.')
            tgt_parts = tgt_value.split('.')

            src_word = src_parts[-2] if len(src_parts) >= 2 else ''
            tgt_word = tgt_parts[-2] if len(tgt_parts) >= 2 else ''

            result.append((src_word, tgt_word))

        return result

    # Define relative path
    #WORK_FOLDER_PATH = "workdir"
    #CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process_test.yml")

    #retrieves the dictionary for models and their sub-models and returns as an ordered list of the full "paths" to the variables
    ordered_vars = process_variables(model_data['tree']['children'])

    #extracts connections for which the target is above the source, in the ordered list of variables
    connections_list = model_data['connections_list']
    result_list = []
    distance_of_BLC = []
    for data in connections_list:
        src_position = ordered_vars.index(data['src'])
        tgt_position = ordered_vars.index(data['tgt'])
        if src_position > tgt_position:
            result_list.append(data)
            distance_of_BLC.append(src_position - tgt_position) #register also how far apart they are


    #extract the BLCs from the variables
    list_of_BLC_in_feedback = extract_BLC(result_list)
    list_of_BLC_in_feedback = [(a, b) for (a, b) in list_of_BLC_in_feedback if 'fastoad_shaper' not in (a, b)] # remove fastoad_shaper from feedback counts


    if INFO:
        print('\n There are', len(list_of_BLC_in_feedback), 'feedback connections \n')

        #Prints the source and target of the feedback loop, along with the distance (in number of BLCs) that separates them
        print("Feedback loops:")
        loop_counter = 1
        loop_set = set()
        for pair in list_of_BLC_in_feedback:
                loop_set.add(pair)
                print(f"{loop_counter}. {pair[0]} -> {pair[1]} | {distance_of_BLC[loop_counter-1]} BLCs")
                loop_counter += 1
        print("\n")
        print('\n Time taken', time.time() - start, 'seconds')

    print('Feedbacks: ', len(list_of_BLC_in_feedback))
    
    return len(list_of_BLC_in_feedback)




#TODO: if info is false, only compute the number, not save every src tgt pair