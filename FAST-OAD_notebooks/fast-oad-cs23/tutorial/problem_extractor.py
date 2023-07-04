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

print(type(model_data))
#print(model_data.keys())
#dict_keys(['tree', 'md5_hash', 'sys_pathnames_list', 'connections_list', 'abs2prom', 'driver', 'design_vars', 'responses', 'declare_partials_list'])
print(str(model_data['connections_list']))

with open('output.txt', 'w') as f:
    f.write(str(model_data['connections_list']))
#print(model_data.values())