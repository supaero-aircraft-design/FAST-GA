"""Test module for the generate_block_analysis function."""
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

import os.path as pth
import os
import pytest
import warnings

import fastoad.api as oad

from fastga.command import api
from fastga.command.unitary_tests.dummy_classes import Disc1, Disc2, Disc3
from fastga import models
from fastga.models import (
    aerodynamics,
    geometry,
    handling_qualities,
    load_analysis,
    loops,
    performances,
)
from fastga.models.weight import cg, mass_breakdown
from fastga.utils.warnings import VariableDescriptionWarning

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_simple_working():

    complete_xml_file = pth.join(pth.dirname(__file__), "data/complete.xml")
    var_inputs = []

    test_generate_block_analysis = api.generate_block_analysis(
        Disc1(), var_inputs, complete_xml_file, overwrite=False
    )
    input_dict = {}
    output_dict = test_generate_block_analysis(input_dict)
    value = output_dict.get("data:geometry:variable_4")[0]
    assert value == pytest.approx(14.0, abs=1e-3)


def test_id_working():

    complete_xml_file = pth.join(pth.dirname(__file__), "data/missing_two_input.xml")
    var_inputs = ["data:geometry:variable_2"]

    # Create a temporary configurator so the modules are registered and their ID are available
    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, "blank.yml"))

    # Test if there were no issues with the configurator
    assert configurator is not None

    test_generate_block_analysis_id = api.generate_block_analysis(
        Disc3(ivc_value=3.0), var_inputs, complete_xml_file, overwrite=False
    )
    input_dict = {"data:geometry:variable_2": ([-5.0, 2.0], "m**2")}
    output_dict = test_generate_block_analysis_id(input_dict)
    value_direct = output_dict.get("data:geometry:variable_4")[0]

    test_generate_block_analysis_id = api.generate_block_analysis(
        "test.dummy_module.disc3",
        var_inputs,
        complete_xml_file,
        overwrite=False,
        options={"ivc_value": 3.0},
    )
    input_dict = {"data:geometry:variable_2": ([-5.0, 2.0], "m**2")}
    output_dict = test_generate_block_analysis_id(input_dict)
    value_id = output_dict.get("data:geometry:variable_4")[0]

    assert value_id == pytest.approx(value_direct, abs=1e-3)

    test_generate_block_analysis_yml = api.generate_block_analysis(
        pth.join(DATA_FOLDER_PATH, "blank.yml"),
        var_inputs,
        complete_xml_file,
        overwrite=False,
    )
    input_dict = {"data:geometry:variable_2": ([-5.0, 2.0], "m**2")}
    output_dict = test_generate_block_analysis_yml(input_dict)
    value_yml = output_dict.get("data:geometry:variable_4")[0]

    assert value_yml == pytest.approx(value_direct, abs=1e-3)


def test_input_vars_working():

    missing_input_xml_file = pth.join(pth.dirname(__file__), "data/missing_one_input.xml")
    var_inputs = ["data:geometry:variable_1"]

    test_generate_block_analysis = api.generate_block_analysis(
        Disc1(), var_inputs, missing_input_xml_file, overwrite=False
    )
    input_dict = {"data:geometry:variable_1": (4.0, None)}
    output_dict = test_generate_block_analysis(input_dict)
    value = output_dict.get("data:geometry:variable_4")[0]
    assert value == pytest.approx(17.0, abs=1e-3)


def test_ivc_working():
    missing_input_xml_file = pth.join(pth.dirname(__file__), "data/missing_one_input.xml")

    var_inputs = []

    test_generate_block_analysis = api.generate_block_analysis(
        Disc2(), var_inputs, missing_input_xml_file, overwrite=False
    )
    input_dict = {}
    output_dict = test_generate_block_analysis(input_dict)
    value = output_dict.get("data:geometry:variable_4")[0]
    assert value == pytest.approx(19.0, abs=1e-3)


def test_supernumerary_inputs():
    """Tests if the various errors implemented in the function handling are caught"""

    xml_file = pth.join(pth.dirname(__file__), "data/supernumerary_inputs.xml")

    var_inputs = ["data:geometry:variable_3", "data:geometry:variable_error"]

    # noinspection PyBroadException
    try:

        test_generate_block_analysis = api.generate_block_analysis(
            Disc1(), var_inputs, xml_file, overwrite=False
        )
        assert test_generate_block_analysis is not None
        function_generated = True
        right_error = False

    except BaseException as error:

        function_generated = False

        if error.args[0] == "The input list contains name(s) out of component/group input list!":
            right_error = True
        else:
            right_error = False

    assert not function_generated
    assert right_error


def test_empty_xml():
    """Tests if the various errors implemented in the function handling are caught"""

    missing_xml = pth.join(pth.dirname(__file__), "data/missing.xml")

    var_inputs = ["data:geometry:variable_3"]

    # First try with no xml file, should fail
    # noinspection PyBroadException
    try:
        # noinspection PyUnusedLocal
        test_generate_block_analysis = api.generate_block_analysis(
            Disc1(), var_inputs, missing_xml, overwrite=False
        )
        function_generated = True
        right_error = False

    except Exception as error:
        print(error.args[0])
        function_generated = False

        if error.args[
            0
        ] == "Input .xml file not found, a default file has been created with default NaN values, but no function is returned!\nConsider defining proper values before second execution!" or (
            "File" in error.args[0] and " is unavailable for reading." in error.args[0]
        ):
            right_error = True
        else:
            right_error = False

    assert not function_generated
    assert right_error

    if os.path.exists(missing_xml):
        os.remove(missing_xml)

    # Then test with empty xml file but var_inputs is contains all the necessary data,
    # should succeed

    var_inputs = [
        "data:geometry:variable_1",
        "data:geometry:variable_2",
        "data:geometry:variable_3",
    ]

    # noinspection PyBroadException
    # noinspection PyUnusedLocal
    try:
        test_generate_block_analysis = api.generate_block_analysis(
            Disc1(), var_inputs, missing_xml, overwrite=False
        )
        function_generated = True
        input_dict = {
            "data:geometry:variable_1": (1.0, None),
            "data:geometry:variable_2": ([3.0, 5.0], "m**2"),
            "data:geometry:variable_3": (2.0, None),
        }
        output_dict = test_generate_block_analysis(input_dict)
        value = output_dict.get("data:geometry:variable_4")[0]

    except Exception as error:

        function_generated = False
        value = 0.0

    assert function_generated
    assert value == pytest.approx(14.0, abs=1e-3)

    if os.path.exists(missing_xml):
        os.remove(missing_xml)


def test_missing_inputs_in_xml():

    missing_inputs_xml_file = pth.join(pth.dirname(__file__), "data/missing_two_input.xml")
    var_inputs = ["data:geometry:variable_2"]

    # noinspection PyBroadException
    try:
        # noinspection PyUnusedLocal
        test_generate_block_analysis = api.generate_block_analysis(
            Disc1(), var_inputs, missing_inputs_xml_file, overwrite=False
        )
        function_generated = True
        right_error = False

    except BaseException as error:

        function_generated = False

        if "The following inputs are missing in .xml file:" in error.args[0]:
            right_error = True
        else:
            right_error = False

    assert not function_generated
    assert right_error


def test_variable_descriptions_auto_gen():

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        api.generate_variables_description(aerodynamics.__path__[0], True)
        api.generate_variables_description(geometry.__path__[0], True)
        api.generate_variables_description(handling_qualities.__path__[0], True)
        api.generate_variables_description(load_analysis.__path__[0], True)
        api.generate_variables_description(loops.__path__[0], True)
        api.generate_variables_description(performances.__path__[0], True)
        api.generate_variables_description(cg.__path__[0], True)
        api.generate_variables_description(mass_breakdown.__path__[0], True)
        api.generate_variables_description(models.__path__[0], True)

        # Check that there are no warnings, which would mean we filled our description correctly
        counter = 0
        if len(w) != 0:
            for warning_message in w:
                if issubclass(warning_message.category, VariableDescriptionWarning):
                    print(warning_message.message)
                    counter += 1

        assert counter == 0
