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

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER

XML_FILE = "cirrus_sr22.xml"


def test_compute_static_margin():
    """ Tests computation of static margin """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeStaticMargin(), input_vars)
    stick_fixed_static_margin = problem["data:handling_qualities:stick_fixed_static_margin"]
    assert stick_fixed_static_margin == pytest.approx(0.0479, rel=1e-2)
    free_elevator_factor = problem["data:aerodynamics:cruise:neutral_point:free_elevator_factor"]
    assert free_elevator_factor == pytest.approx(0.7217, rel=1e-2)
    stick_free_static_margin = problem["data:handling_qualities:stick_free_static_margin"]
    assert stick_free_static_margin == pytest.approx(-0.0253, rel=1e-2)


def test_compute_to_rotation_limit():
    """ Tests computation of static margin """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeTORotationLimitGroup(propulsion_id=ENGINE_WRAPPER), input_vars)
    x_cg_rotation_limit = problem["data:handling_qualities:to_rotation_limit:x"]
    assert x_cg_rotation_limit == pytest.approx(1.9355, rel=1e-2)
    x_cg_ratio_rotation_limit = problem["data:handling_qualities:to_rotation_limit:MAC_position"]
    assert x_cg_ratio_rotation_limit == pytest.approx(-0.3451, rel=1e-2)


def test_compute_balked_landing():
    """ Tests computation of static margin """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBalkedLandingLimit(propulsion_id=ENGINE_WRAPPER), input_vars)
    x_cg_balked_landing_limit = problem["data:handling_qualities:balked_landing_limit:x"]
    assert x_cg_balked_landing_limit == pytest.approx(2.0738, rel=1e-2)
    x_cg_ratio_balked_landing_limit = problem[
        "data:handling_qualities:balked_landing_limit:MAC_position"
    ]
    assert x_cg_ratio_balked_landing_limit == pytest.approx(-0.23, rel=1e-2)


def test_update_vt_area():
    """ Tests computation of the vertical tail area """

    # Research independent input value in .xml file
    input_vars = get_indep_var_comp(
        list_inputs(UpdateVTArea(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    input_vars.add_output("data:weight:aircraft:OWE", 1039.139, units="kg")
    input_vars.add_output("data:weight:aircraft:payload", 355.0, units="kg")
    input_vars.add_output("data:aerodynamics:fuselage:cruise:CnBeta", -0.0599)
    input_vars.add_output("data:aerodynamics:rudder:low_speed:Cy_delta_r", 1.3536, units="rad**-1")
    input_vars.add_output(
        "data:aerodynamics:vertical_tail:low_speed:CL_alpha", 2.634, units="rad**-1"
    )
    input_vars.add_output("data:geometry:cabin:length", 2.86, units="m")
    input_vars.add_output("data:geometry:fuselage:front_length", 1.559, units="m")
    input_vars.add_output("data:geometry:fuselage:rear_length", 3.15, units="m")
    input_vars.add_output("data:geometry:fuselage:maximum_height", 1.41, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateVTArea(propulsion_id=ENGINE_WRAPPER), input_vars)
    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(1.751, abs=1e-2)  # old-version obtained value 2.4m²


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
    assert ht_area == pytest.approx(3.877, abs=1e-2)
