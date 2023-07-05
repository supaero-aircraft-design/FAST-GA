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

def extract_BLC(data): #extracts Bottom level components for feedback loops
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

#retrieves the dictionary for models and their sub-models
ordered_vars = process_variables(model_data['tree']['children'])

connections_list = model_data['connections_list'] #data_list
result_list = []
for data in connections_list:
    src_position = ordered_vars.index(data['src'])
    tgt_position = ordered_vars.index(data['tgt'])
    if src_position > tgt_position:
        result_list.append(data)


print('There are', len(result_list), 'feedback connections')

print('\n They are: \n')
#print(result_list)

print(extract_BLC(result_list))
"""src_words = []
tgt_words = []

for entry in data:
    src_value = entry['src']
    tgt_value = entry['tgt']

    src_word = src_value.rsplit('.', 1)[0].rsplit('.', 1)[-1]
    tgt_word = tgt_value.rsplit('.', 1)[0].rsplit('.', 1)[-1]

    src_words.append(src_word)
    tgt_words.append(tgt_word)

print("bottom level component source of feedback: ", src_words)
print("bottom level component target of feedback", tgt_words)"""

#print(result_list)
