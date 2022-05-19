"""
Test module for Overall Aircraft Design process
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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
from numpy.testing import assert_allclose

from fastga.command import api
from fastga.models.geometry.geometry import GeometryFixedTailDistance as Geometry
from fastga.models.aerodynamics.aerodynamics import Aerodynamics
import fastga.notebooks.tutorial.data as data

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1 : len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")

AIRCRAFT_ID = ["be76", "sr22"]
MDA_WING_POSITION = False


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_analysis_mode():
    """
    Test the analysis mode is still working.
    """

    # Copy used file
    os.mkdir(RESULTS_FOLDER_PATH)
    shutil.copy(
        pth.join(data.__path__[0], "reference_aircraft.xml"),
        pth.join(RESULTS_FOLDER_PATH, "geometry_long_wing.xml"),
    )
    var_inputs = [
        "data:geometry:wing:area",
        "data:geometry:wing:aspect_ratio",
        "data:geometry:wing:taper_ratio",
    ]

    # Declare geometry function
    compute_geometry = api.generate_block_analysis(
        Geometry(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
        var_inputs,
        str(pth.join(RESULTS_FOLDER_PATH, "geometry_long_wing.xml")),
        True,
    )

    # Compute long-wing aircraft
    inputs_dict = {
        "data:geometry:wing:area": (21.66, "m**2"),
        "data:geometry:wing:aspect_ratio": (9.332, None),
        "data:geometry:wing:taper_ratio": (0.77, None),
    }
    outputs_dict = compute_geometry(inputs_dict)
    assert_allclose(outputs_dict["data:geometry:wing:tip:y"][0], 7.108, atol=1e-3)

    # Compute aerodynamics for both aircraft
    compute_aero = api.generate_block_analysis(
        Aerodynamics(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            use_openvsp=True,
            compute_mach_interpolation=False,
            compute_slipstream_low_speed=False,
            compute_slipstream_cruise=False,
            result_folder_path="D:/tmp",
        ),
        [],
        str(pth.join(RESULTS_FOLDER_PATH, "geometry_long_wing.xml")),
        True,
    )

    compute_aero({})
