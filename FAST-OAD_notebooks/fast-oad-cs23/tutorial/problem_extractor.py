from fastoad.io.configuration import FASTOADProblemConfigurator
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.utils.om_warnings import issue_warning
import os.path as pth
import os
import ast

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
            variables.extend(process_variables(child['children'], child_prefix))
    return variables



def get_dict_depth(dictionary): 
    if isinstance(dictionary, dict):
        if len(dictionary) == 0:
            return 1
        else:
            return 1 + max(get_dict_depth(value) for value in dictionary.values())
    else:
        return 0
    




# Define relative path
DATA_FOLDER_PATH = "data"
WORK_FOLDER_PATH = "workdir"
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process_test.yml")

# Used for test purposes only
_PROBLEM_CONFIGURATOR = None

#Setup of the problem
conf = FASTOADProblemConfigurator(CONFIGURATION_FILE)
conf._set_configuration_modifier(_PROBLEM_CONFIGURATOR)
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
#dict_keys(['tree', 'md5_hash', 'sys_pathnames_list', 'connections_list', 'abs2prom', 'driver', 'design_vars', 'responses', 'declare_partials_list'])

#retrieves the dictionary for models and their sub-models
ordered_vars = process_variables(model_data['tree']['children'])


connections_list = model_data['connections_list'] #data_list
result_list = []
for data in connections_list:
    src_position = ordered_vars.index(data['src'])
    #print(data['src'], src_position)
    tgt_position = ordered_vars.index(data['tgt'])
    #print(data['tgt'], tgt_position)
    if src_position > tgt_position:
        result_list.append(data)
        #print('OK!')


print(len(result_list))
"""

with open('output_tree.txt', 'w') as f:
    f.write(str(model_data['tree']))
#print(model_data.values())"""
