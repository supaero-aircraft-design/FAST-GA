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
import logging
import os.path as pth
from shutil import rmtree
from platform import system

import openmdao.api as om
import pytest
from numpy.testing import assert_allclose

from fastoad.io.configuration.configuration import FASTOADProblemConfigurator

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1 : len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")

AIRCRAFT_ID = ["sr22", "be76"]


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_oad_process_vlm(cleanup):
    """
    Test the overall aircraft design process with wing positioning under VLM method.
    """

    logging.basicConfig(level=logging.WARNING)

    for aircraft_id in AIRCRAFT_ID:
        # Define used files depending on options
        xml_file_name = "input_" + aircraft_id + ".xml"
        process_file_name = "oad_process_" + aircraft_id + ".yml"

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

        if aircraft_id == "sr22":
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 263.0, atol=1)
            assert_allclose(
                problem["data:handling_qualities:stick_fixed_static_margin"], 0.10, atol=1e-2
            )
            # noinspection PyTypeChecker
            assert_allclose(
                problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1627.0, atol=1
            )
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1008.0, atol=1)
        else:
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 246.0, atol=1)
            assert_allclose(
                problem["data:handling_qualities:stick_fixed_static_margin"], 0.15, atol=1e-2
            )
            # noinspection PyTypeChecker
            assert_allclose(
                problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1767.0, atol=1
            )
            # noinspection PyTypeChecker
            assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1131.0, atol=1)


@pytest.mark.skipif(system() != "Windows", reason="OPENVSP is windows dependent platform")
def _test_oad_process_openvsp():
    """
    Test the overall aircraft design process only on Cirrus with wing positioning under OpenVSP method.
    """

    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "oad_process_sr22_openvsp.yml"

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

    # Check values
    # noinspection PyTypeChecker
    assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 252.0, atol=1)
    assert_allclose(problem["data:handling_qualities:stick_fixed_static_margin"], 0.10, atol=1e-2)
    # noinspection PyTypeChecker
    assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1613.0, atol=1)
    # noinspection PyTypeChecker
    assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1006.0, atol=1)


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
