from fastoad.io.configuration import FASTOADProblemConfigurator
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.utils.om_warnings import issue_warning
import os.path as pth
import os

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

#print(type(model_data))
#print(model_data.keys())
#dict_keys(['tree', 'md5_hash', 'sys_pathnames_list', 'connections_list', 'abs2prom', 'driver', 'design_vars', 'responses', 'declare_partials_list'])
#print(str(model_data['design_vars']))


#obtain from the dictionary abs2prom the ordered list of variables
ordered_vars = [] #string1_list
for pair in model_data['abs2prom'].values():
    ordered_vars.extend(pair.keys())

connections_list = model_data['connections_list'] #data_list
result_list = []
for data in connections_list:
    src_position = ordered_vars.index(data['src'])
    print(data['src'], src_position)
    tgt_position = ordered_vars.index(data['tgt'])
    print(data['tgt'], tgt_position)
    if src_position > tgt_position:
        result_list.append(data)
        print('OK!')

#print(result_list)#DOESNT WORK



"""with open('output_abs2prom.txt', 'w') as f:
    f.write(str(model_data['abs2prom']))
#print(model_data.values())"""
