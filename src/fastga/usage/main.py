import os.path as pth
from fastoad import api as api_cs25
from fastga.command import api as api_cs23
import logging
from fastoad.gui import VariableViewer
import shutil

# Define relative path
DATA_FOLDER_PATH = "data"
WORK_FOLDER_PATH = "workdir"


# Define files
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process.yml")
SOURCE_FILE = pth.join(DATA_FOLDER_PATH, "fuel_cell_aircraft.xml")
HQ_CONFIG_FILE = pth.join(WORK_FOLDER_PATH, "hq_process.yml")
SOURCE_OUTPUT_FC = pth.join(DATA_FOLDER_PATH,"problem_outputs_fuel_cell_150NM_mda.xml")
SOURCE_OUTPUT_IC = pth.join(DATA_FOLDER_PATH,"problem_outputs_Beechcraft_800nm_mda.xml")
INPUT_FILE = pth.join(WORK_FOLDER_PATH, 'problem_inputs.xml')

# Generation d'un nouveau fichier d'input à partir du process décrit dans CONFIGURATION_FILE
# SOURCE_FILE est utilise pour attribue les valeurs aux variables trouvees dans le process
# api_cs25.generate_inputs(CONFIGURATION_FILE, SOURCE_OUTPUT_FC, overwrite=True)

# Liste les modules accessibles pour fast-ga
# api_cs25.list_modules(CONFIGURATION_FILE, force_text_output=True)

# Liste les variables du processus
# api_cs25.list_variables(CONFIGURATION_FILE, force_text_output=True)

#Diagramme N2
# N2_FILE = pth.join(WORK_FOLDER_PATH, "n2.html")
# api_cs25.write_n2(CONFIGURATION_FILE, N2_FILE, overwrite=True)

# MDA
eval_problem = api_cs25.evaluate_problem(CONFIGURATION_FILE, overwrite=True)