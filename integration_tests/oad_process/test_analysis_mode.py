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

import pathlib
import shutil
from shutil import rmtree

import pytest
from numpy.testing import assert_allclose

from fastga.command import api
from fastga.models.aerodynamics.aerodynamics import Aerodynamics
from fastga.models.geometry.geometry import GeometryFixedTailDistance as Geometry
from fastga.notebooks.tutorial import data

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
WORKDIR_FOLDER_PATH = pathlib.Path(__file__).parent / "workdir"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree(WORKDIR_FOLDER_PATH, ignore_errors=True)


def test_analysis_mode():
    """
    Test the analysis mode is still working.
    """

    # Copy used file
    if not RESULTS_FOLDER_PATH.exists():
        RESULTS_FOLDER_PATH.mkdir()

    shutil.copy(
        pathlib.Path(data.__path__[0]) / "reference_aircraft.xml",
        RESULTS_FOLDER_PATH / "geometry_long_wing.xml",
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
        RESULTS_FOLDER_PATH / "geometry_long_wing.xml",
        overwrite=True,
    )

    # Compute long-wing aircraft
    inputs_dict = {
        "data:geometry:wing:area": (21.66, "m**2"),
        "data:geometry:wing:aspect_ratio": (9.332, "unitless"),
        "data:geometry:wing:taper_ratio": (0.77, "unitless"),
    }
    outputs_dict = compute_geometry(inputs_dict)
    assert_allclose(outputs_dict["data:geometry:wing:tip:y"][0], 7.108, atol=1e-3)

    var_inputs = [
        "data:geometry:wing:root:virtual_chord",
        "data:geometry:landing_gear:height",
        "data:geometry:wing:wet_area",
        "data:geometry:wing:MAC:length",
        "data:geometry:vertical_tail:root:chord",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:fuselage:volume",
        "data:geometry:wing:MAC:leading_edge:x:local",
        "data:geometry:vertical_tail:tip:chord",
        "data:geometry:fuselage:average_depth",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:wing:tip:chord",
        "data:geometry:fuselage:length",
        "data:geometry:vertical_tail:wet_area",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:horizontal_tail:span",
        "data:geometry:wing:tip:y",
        "data:geometry:wing:root:z",
        "data:geometry:horizontal_tail:z:from_wingMAC25",
        "data:geometry:horizontal_tail:root:chord",
        "data:geometry:wing:root:y",
        "data:aerodynamics:horizontal_tail:efficiency",
        "data:geometry:fuselage:wet_area",
        "data:geometry:horizontal_tail:MAC:at25percent:x:local",
        "data:geometry:wing:span",
        "data:geometry:vertical_tail:MAC:z",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:wing:root:chord",
        "data:geometry:horizontal_tail:wet_area",
        "data:geometry:vertical_tail:span",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:wing:sweep_0",
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
    ]

    inputs_dict = {}
    for input_name in var_inputs:
        inputs_dict[input_name] = outputs_dict[input_name]

    # Compute aerodynamics for both aircraft
    compute_aero = api.generate_block_analysis(
        Aerodynamics(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            use_openvsp=True,
            compute_mach_interpolation=False,
            compute_slipstream_low_speed=False,
            compute_slipstream_cruise=False,
            result_folder_path=WORKDIR_FOLDER_PATH.as_posix(),
        ),
        var_inputs,
        RESULTS_FOLDER_PATH / "geometry_long_wing.xml",
        overwrite=True,
    )

    compute_aero(inputs_dict)
