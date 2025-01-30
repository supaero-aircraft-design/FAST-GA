"""
Test module for tail weight services registry.
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
import pytest
import os.path as pth
import fastoad.api as oad
from ..a_airframe.constants import (
    SUBMODEL_TAIL_MASS,
    SUBMODEL_HTP_MASS,
    SUBMODEL_VTP_MASS,
)
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..a_airframe.sum import AirframeWeight

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
XML_FILE = "cirrus_sr22.xml"


def test_tail_mass_registry():
    """Tests tail mass submodel registry with integration ."""
    process_file_name = "dummy_conf.yml"
    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()
    problem.setup()

    assert (
        oad.RegisterSubmodel.active_models["submodel.weight.mass.airframe.htp"]
        == "fastga.submodel.weight.mass.airframe.htp.torenbeek"
    )
    assert (
        oad.RegisterSubmodel.active_models["submodel.weight.mass.airframe.vtp"]
        == "fastga.submodel.weight.mass.airframe.vtp.legacy"
    )


def test_tail_weight_compatibility(_reset_tail_submodel_registry):
    """Tests tail weight calculation submodel compatibility."""
    # Research independent input value in .xml file

    system = AirframeWeight()

    oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] = (
        "fastga.submodel.weight.mass.airframe.tail.gd"
    )

    ivc = get_indep_var_comp(list_inputs(system), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    run_system(system, ivc)
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS]
        == "fastga.submodel.weight.mass.airframe.htp.gd"
    )
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS]
        == "fastga.submodel.weight.mass.airframe.vtp.gd"
    )


def test_tail_weight_compatibility_overwrite_htp(_reset_tail_submodel_registry):
    oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] = (
        "fastga.submodel.weight.mass.airframe.tail.gd"
    )
    oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
        "fastga.submodel.weight.mass.airframe.htp.legacy"
    )
    system = AirframeWeight()
    ivc = get_indep_var_comp(list_inputs(system), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    run_system(system, ivc)
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS]
        == "fastga.submodel.weight.mass.airframe.htp.legacy"
    )
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS]
        == "fastga.submodel.weight.mass.airframe.vtp.gd"
    )


def test_tail_weight_compatibility_overwrite_vtp(_reset_tail_submodel_registry):
    oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] = (
        "fastga.submodel.weight.mass.airframe.tail.gd"
    )
    oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
        "fastga.submodel.weight.mass.airframe.vtp.legacy"
    )
    system = AirframeWeight()
    ivc = get_indep_var_comp(list_inputs(system), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    run_system(system, ivc)
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS]
        == "fastga.submodel.weight.mass.airframe.htp.gd"
    )
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS]
        == "fastga.submodel.weight.mass.airframe.vtp.legacy"
    )


def test_tail_weight_compatibility_only_htp(_reset_tail_submodel_registry):
    oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
        "fastga.submodel.weight.mass.airframe.htp.gd"
    )
    system = AirframeWeight()
    ivc = get_indep_var_comp(list_inputs(system), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    run_system(system, ivc)
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS]
        == "fastga.submodel.weight.mass.airframe.htp.gd"
    )
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS]
        == "fastga.submodel.weight.mass.airframe.vtp.legacy"
    )


def test_tail_weight_compatibility_only_vtp(_reset_tail_submodel_registry):
    oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
        "fastga.submodel.weight.mass.airframe.vtp.gd"
    )
    system = AirframeWeight()
    ivc = get_indep_var_comp(list_inputs(system), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    run_system(system, ivc)
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS]
        == "fastga.submodel.weight.mass.airframe.htp.legacy"
    )
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS]
        == "fastga.submodel.weight.mass.airframe.vtp.gd"
    )


def test_tail_weight_compatibility_both(_reset_tail_submodel_registry):
    oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
        "fastga.submodel.weight.mass.airframe.htp.gd"
    )
    oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
        "fastga.submodel.weight.mass.airframe.vtp.gd"
    )
    system = AirframeWeight()
    ivc = get_indep_var_comp(list_inputs(system), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    run_system(system, ivc)
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS]
        == "fastga.submodel.weight.mass.airframe.htp.gd"
    )
    assert (
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS]
        == "fastga.submodel.weight.mass.airframe.vtp.gd"
    )


@pytest.fixture
def _reset_tail_submodel_registry():
    oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = None
    oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = None
    oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] = (
        "fastga.submodel.weight.mass.airframe.tail.legacy"
    )
