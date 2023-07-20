import warnings
warnings.filterwarnings(action="ignore")

import os.path as pth
import os
import openmdao.api as om
from fastoad import api as api_cs25
from fastga.command import api as api_cs23
import subprocess
import time


# Define relative path
DATA_FOLDER_PATH = "data"
WORK_FOLDER_PATH = "workdir"
SRC_FOLDER_PATH = "src/fastga/source_files"
times_module = []

for i in range(15):
    print('Loop: ', i)
    try:
        os.remove(WORK_FOLDER_PATH, "problem_outputs.xml")
    except:
        print('nothing removed')
    # Define files

    CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process_test.yml")
    SOURCE_FILE = pth.join(DATA_FOLDER_PATH, "beechcraft_76.xml")

    api_cs25.generate_inputs(CONFIGURATION_FILE, SOURCE_FILE, overwrite=True)

    starting = time.time()
    eval_problem = api_cs25.evaluate_problem(CONFIGURATION_FILE, overwrite=True)
    #optim_problem = api_cs25.optimize_problem(CONFIGURATION_FILE, overwrite=True)


    

    times_module.append(time.time() - starting)

print('\n Sample problem without opt ran in ', sum(times_module)/len(times_module) , ' seconds \n')