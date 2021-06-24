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
from shutil import rmtree
from fastga.command import api

import openmdao.api as om
import pytest
from numpy.testing import assert_allclose

from fastoad.io.configuration.configuration import FASTOADProblemConfigurator
from fastga.models.aerodynamics.components.compute_propeller_aero import ComputePropellerPerformance

# from fastoad import api

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)

AIRCRAFT_ID = "sr22"  # "be76"
MDA_WING_POSITION = True


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_propeller(cleanup):
    """
    Test the execution of the geometry block
    """

    # Define used files
    xml_file_name = "input_be76.xml"

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    compute_propeller = api.generate_block_analysis(
        ComputePropellerPerformance(),
        [],
        ref_inputs,
        True,
    )

    test = 1.0

    outputs_dict = compute_propeller({})

    print(outputs_dict)
    # Create problems with inputs
