"""
Test module for Overall Aircraft Design process
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import os.path as pth
import shutil
from shutil import rmtree

import pytest
from fastoad import api as api_cs25

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1 : len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")

AIRCRAFT_ID = "be76"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_oad_process(cleanup):
    """
    Test the overall aircraft design process without wing positioning.
    """

    xml_file_name = "input_" + AIRCRAFT_ID + "_wing_pos.xml"
    process_file_name = "mdo_process.yml"

    INPUT_FILE_PATH = pth.join(DATA_FOLDER_PATH, xml_file_name)
    PROCESS_FILE_PATH = pth.join(DATA_FOLDER_PATH, process_file_name)

    api_cs25.generate_inputs(PROCESS_FILE_PATH, INPUT_FILE_PATH, overwrite=True)

    api_cs25.optimization_viewer(PROCESS_FILE_PATH)

    test2 = api_cs25.variable_viewer(INPUT_FILE_PATH)

    test = 12
