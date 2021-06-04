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

import openmdao.api as om
import pytest
from numpy.testing import assert_allclose

from fastoad.io.configuration.configuration import FASTOADProblemConfigurator

from fastga.command import api
from fastga.models.geometry.geometry import GeometryFixedTailDistance as Geometry
from fastga.models.aerodynamics.aerodynamics_high_speed import AerodynamicsHighSpeed
from fastga.models.aerodynamics.aerodynamics_low_speed import AerodynamicsLowSpeed
from fastga.models.aerodynamics.aerodynamics import Aerodynamics
import fastga.notebooks.tutorial.data as data
# from fastoad import api

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1:len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")

AIRCRAFT_ID = "be76"  # "sr22"
MDA_WING_POSITION = True


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_oad_process(cleanup):
    """
    Test the overall aircraft design process without wing positioning.
    """

    # Define used files depending on options
    if MDA_WING_POSITION:
        xml_file_name = 'input_' + AIRCRAFT_ID + '_wing_pos.xml'
        process_file_name = 'oad_process_' + AIRCRAFT_ID + '_wing_pos.yml'
    else:
        xml_file_name = 'input_' + AIRCRAFT_ID + '.xml'
        process_file_name = 'oad_process_' + AIRCRAFT_ID + '.yml'

    configurator = FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.run_model()
    problem.write_outputs()

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)
    om.view_connections(
        problem, outfile=pth.join(RESULTS_FOLDER_PATH, "connections.html"), show_browser=False
    )
    om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "n2.html"), show_browser=False)

    # Check that weight-performances loop correctly converged
    _check_weight_performance_loop(problem)

    if MDA_WING_POSITION:
        if AIRCRAFT_ID == "sr22":
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 250, atol=1)
            assert_allclose(problem["data:handling_qualities:stick_fixed_static_margin"], 0.10, atol=1e-2)
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1644, atol=1)
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1039, atol=1)
        else:
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 214., atol=1)
            assert_allclose(problem["data:handling_qualities:stick_fixed_static_margin"], 0.15, atol=1e-2)
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1702., atol=1)
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1098., atol=1)


def _check_weight_performance_loop(problem):
    assert_allclose(
        problem["data:weight:aircraft:OWE"],
        problem["data:weight:airframe:mass"]
        + problem["data:weight:propulsion:mass"]
        + problem["data:weight:systems:mass"]
        + problem["data:weight:furniture:mass"],
        rtol=5e-2,
    )
    assert_allclose(
        problem["data:weight:aircraft:MZFW"],
        problem["data:weight:aircraft:OWE"] + problem["data:weight:aircraft:max_payload"],
        rtol=5e-2,
    )
    assert_allclose(
        problem["data:weight:aircraft:MTOW"],
        problem["data:weight:aircraft:OWE"]
        + problem["data:weight:aircraft:payload"]
        + problem["data:mission:sizing:fuel"],
        rtol=5e-2,
    )


def _test_notebooks(cleanup):
    # Copy used file
    os.mkdir(RESULTS_FOLDER_PATH)
    shutil.copy(pth.join(data.__path__[0], 'reference_aircraft.xml'),
                pth.join(RESULTS_FOLDER_PATH, 'geometry_long_wing.xml'))
    var_inputs = ["data:geometry:wing:area", "data:geometry:wing:aspect_ratio", "data:geometry:wing:taper_ratio"]

    # Declare geometry function
    compute_geometry = api.generate_block_analysis(
        Geometry(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
        var_inputs,
        str(pth.join(RESULTS_FOLDER_PATH, 'geometry_long_wing.xml')),
        True,
    )

    # Compute long-wing aircraft
    inputs_dict = {"data:geometry:wing:area": (21.66, "m**2"), "data:geometry:wing:aspect_ratio": (9.332, None),
                   "data:geometry:wing:taper_ratio": (0.77, None)}
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
            result_folder_path='D:/tmp',
        ),
        [],
        str(pth.join(RESULTS_FOLDER_PATH, 'geometry_long_wing.xml')),
        True,
    )

    # compute_aero_LS = api.generate_block_analysis(
    #     AerodynamicsLowSpeed(
    #         propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
    #         use_openvsp=True,
    #         compute_slipstream=False,
    #     ),
    #     [],
    #     str(pth.join(RESULTS_FOLDER_PATH, 'geometry_long_wing.xml')),
    #     True,
    # )
    # compute_aero_HS = api.generate_block_analysis(
    #     AerodynamicsHighSpeed(
    #         propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
    #         use_openvsp=True,
    #         compute_mach_interpolation=False,
    #         compute_slipstream=False,
    #     ),
    #     [],
    #     str(pth.join(RESULTS_FOLDER_PATH, 'geometry_long_wing.xml')),
    #     True,
    # )
    # compute_aero_LS({})
    # compute_aero_HS({})
    compute_aero({})


