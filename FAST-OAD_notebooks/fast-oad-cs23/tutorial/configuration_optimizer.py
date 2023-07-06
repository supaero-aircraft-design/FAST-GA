import os.path as pth
import os
import sys
import openmdao.api as om
from fastoad import api as api_cs25
from fastga.command import api as api_cs23
import yaml
import shutil
import time

# Define relative path
DATA_FOLDER_PATH = "data"
WORK_FOLDER_PATH = "workdir"

# Remove work folder
#shutil.rmtree(WORK_FOLDER_PATH, ignore_errors=True)

# Define file
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process.yml")


#Define optimizatin level: (assumes presence of aircraft_sizing, with modules defined)
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
#
# LVL 2: Shuffles order of all sub-modules in all the modules of the CONFIG file
# LVL 3: etc 
# LVL 4: etc
start = time.time()

Optimization_level = 1

if Optimization_level == 1:
    with open(CONFIGURATION_FILE, 'r') as file:
        # Load the YAML contents
        yaml_data = yaml.safe_load(file)

        # Get the structure under 'model: aircraft_sizing'
        try:
            yaml_data['model']['aircraft_sizing']
        except:
            sys.exit("\n Error: Configuration optimizer assumes existence of aircraft_sizing inside model .yml config file \n")

        yaml_data['model']['aircraft_sizing'].pop('nonlinear_solver', None)
        yaml_data['model']['aircraft_sizing'].pop('linear_solver', None)
        aircraft_sizing_data = yaml_data['model']['aircraft_sizing']

        


        #print(aircraft_sizing_data)

#elif Optimization_level == 2:
#elif Optimization_level == 3:
#elif Optimization_level == 4:
#elif Optimization_level == 5:
else:
    print('Not possible sry')

print('\n Time taken', time.time() - start, 'seconds')