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

import os.path as pth
import pandas as pd
import numpy as np
from openmdao.core.component import Component
import pytest
from typing import Union

from fastoad.io import VariableIO
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint
from fastoad.model_base.propulsion import IOMPropulsionWrapper
from fastoad.constants import EngineSetting

from ..compute_static_margin import ComputeStaticMargin
from ..tail_sizing.update_vt_area import UpdateVTArea
from ..tail_sizing.update_ht_area import UpdateHTArea
from ..tail_sizing.compute_to_rotation_limit import ComputeTORotationLimitGroup
from ..tail_sizing.compute_balked_landing_limit import ComputeBalkedLandingLimit

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

XML_FILE = "beechcraft_76.xml"


def test_update_vt_area():
    """ Tests computation of the vertical tail area """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(UpdateVTArea(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.364924)
    ivc.add_output("data:weight:aircraft:OWE", 1125.139, units="kg")
    ivc.add_output("data:weight:aircraft:payload", 390.0, units="kg")
    ivc.add_output("data:aerodynamics:fuselage:cruise:CnBeta", -0.0599)
    ivc.add_output("data:aerodynamics:rudder:low_speed:Cy_delta_r", 1.6412, units="rad**-1")
    ivc.add_output("data:aerodynamics:vertical_tail:low_speed:CL_alpha", 2.9967, units="rad**-1")
    ivc.add_output("data:geometry:cabin:length", 3.096, units="m")
    ivc.add_output("data:geometry:fuselage:front_length", 1.87, units="m")
    ivc.add_output("data:geometry:fuselage:rear_length", 3.55, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_height", 1.3378, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateVTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(2.44, abs=1e-2)  # old-version obtained value 2.4m²


def test_update_ht_area():
    """ Tests computation of the horizontal tail area """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(UpdateHTArea(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(UpdateHTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    ht_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    assert ht_area == pytest.approx(3.843, abs=1e-2)  # old-version obtained value 3.9m²


def test_compute_static_margin():
    """ Tests computation of static margin """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.576)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeStaticMargin(), input_vars)
    stick_fixed_static_margin = problem["data:handling_qualities:stick_fixed_static_margin"]
    assert stick_fixed_static_margin == pytest.approx(0.206, rel=1e-2)
    stick_free_static_margin = problem["data:handling_qualities:stick_free_static_margin"]
    assert stick_free_static_margin == pytest.approx(0.147, rel=1e-2)


def test_compute_to_rotation_limit():
    """ Tests the computation of the forward most possible CG location for the TO rotation in case HTP area is fixed"""

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:geometry:horizontal_tail:area", 3.78)

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeTORotationLimitGroup(propulsion_id=ENGINE_WRAPPER), input_vars)
    to_rotation_limit = problem["data:handling_qualities:to_rotation_limit:x"]
    assert to_rotation_limit == pytest.approx(2.99, rel=1e-2)
    to_rotation_limit_ratio = problem["data:handling_qualities:to_rotation_limit:MAC_position"]
    assert to_rotation_limit_ratio == pytest.approx(-0.0419, rel=1e-2)


def test_balked_landing_limit():
    """ Tests the computation of the forward most possible CG location for a balked landing in case HTP area is fixed"""

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:geometry:horizontal_tail:area", 3.78)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBalkedLandingLimit(propulsion_id=ENGINE_WRAPPER), input_vars)
    balked_landing_limit = problem["data:handling_qualities:balked_landing_limit:x"]
    assert balked_landing_limit == pytest.approx(3.43, rel=1e-2)
    balked_landing_limit_ratio = problem[
        "data:handling_qualities:balked_landing_limit:MAC_position"
    ]
    assert balked_landing_limit_ratio == pytest.approx(0.24, rel=1e-2)
