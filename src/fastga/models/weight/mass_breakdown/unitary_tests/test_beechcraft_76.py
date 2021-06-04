"""
Test module for mass breakdown functions
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

from ..a_airframe import (
    ComputeTailWeight,
    ComputeFlightControlsWeight,
    ComputeFuselageWeight,
    ComputeFuselageWeightRaymer,
    ComputeWingWeight,
    ComputeLandingGearWeight,
)
from ..b_propulsion import (
    ComputeOilWeight,
    ComputeFuelLinesWeight,
    ComputeEngineWeight,
    ComputeUnusableFuelWeight,
)
from ..c_systems import (
    ComputeLifeSupportSystemsWeight,
    ComputeNavigationSystemsWeight,
    ComputePowerSystemsWeight,
)
from ..d_furniture import (
    ComputePassengerSeatsWeight,
)
from ..mass_breakdown import MassBreakdown, ComputeOperatingWeightEmpty
from ..payload import ComputePayload

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

XML_FILE = "beechcraft_76.xml"
ENGINE_WRAPPER = "test.wrapper.mass_breakdown.beechcraft.dummy_engine"


class DummyEngine(AbstractFuelPropulsion):

    def __init__(self,
                 max_power: float,
                 design_altitude: float,
                 design_speed: float,
                 fuel_type: float,
                 strokes_nb: float,
                 prop_layout: float,
                 ):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude = design_altitude
        self.design_speed = design_speed
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        flight_points.thrust = 3500.0
        flight_points.sfc = 0.0

    def compute_weight(self) -> float:
        return 562.83 / 2.0

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        return 0.0

    def get_consumed_mass(self, flight_point: FlightPoint, time_step: float) -> float:
        return 0.0


class DummyEngineWrapper(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:IC_engine:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:layout", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "design_altitude": inputs["data:mission:sizing:main_route:cruise:altitude"],
            "design_speed": inputs["data:TLAR:v_cruise"],
            "fuel_type": inputs["data:propulsion:IC_engine:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:layout"]
        }

        return DummyEngine(**engine_params)


# RegisterPropulsion(ENGINE_WRAPPER)(DummyEngineWrapper)


def test_compute_payload():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePayload()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePayload(), ivc)
    assert problem["data:weight:aircraft:payload"] == pytest.approx(350.0, abs=1e-2)
    assert problem["data:weight:aircraft:max_payload"] == pytest.approx(410.0, abs=1e-2)

    # Run problem and check obtained value(s) is/(are) correct
    ivc = get_indep_var_comp(list_inputs(ComputePayload()), __file__, XML_FILE)
    ivc.add_output("settings:weight:aircraft:payload:design_mass_per_passenger", 1.0, units="kg")
    ivc.add_output("settings:weight:aircraft:payload:max_mass_per_passenger", 2.0, units="kg")
    problem = run_system(ComputePayload(), ivc)
    assert problem["data:weight:aircraft:payload"] == pytest.approx(34.0, abs=0.1)
    assert problem["data:weight:aircraft:max_payload"] == pytest.approx(58.0, abs=0.1)


def test_compute_wing_weight():
    """ Tests wing weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingWeight(), ivc)
    weight_a1 = problem.get_val("data:weight:airframe:wing:mass", units="kg")
    assert weight_a1 == pytest.approx(217.459, abs=1e-2)  # difference because of integer conversion error


def test_compute_fuselage_weight():
    """ Tests fuselage weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeight(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass", units="kg")
    assert weight_a2 == pytest.approx(158.493, abs=1e-2)


def test_compute_fuselage_weight_raymer():
    """ Tests fuselage weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageWeightRaymer()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeightRaymer(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass_raymer", units="kg")
    assert weight_a2 == pytest.approx(186.93, abs=1e-2)


def test_compute_empennage_weight():
    """ Tests empennage weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeTailWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTailWeight(), ivc)
    weight_a31 = problem.get_val("data:weight:airframe:horizontal_tail:mass", units="kg")
    assert weight_a31 == pytest.approx(24.94, abs=1e-2)
    weight_a32 = problem.get_val("data:weight:airframe:vertical_tail:mass", units="kg")
    assert weight_a32 == pytest.approx(19.24, abs=1e-2)


def test_compute_flight_controls_weight():
    """ Tests flight controls weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFlightControlsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFlightControlsWeight(), ivc)
    weight_a4 = problem.get_val("data:weight:airframe:flight_controls:mass", units="kg")
    assert weight_a4 == pytest.approx(34.08, abs=1e-2)


def test_compute_landing_gear_weight():
    """ Tests landing gear weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLandingGearWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLandingGearWeight(), ivc)
    weight_a51 = problem.get_val("data:weight:airframe:landing_gear:main:mass", units="kg")
    assert weight_a51 == pytest.approx(59.32, abs=1e-2)
    weight_a52 = problem.get_val("data:weight:airframe:landing_gear:front:mass", units="kg")
    assert weight_a52 == pytest.approx(24.11, abs=1e-2)


def test_compute_oil_weight():
    """ Tests engine weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeOilWeight(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOilWeight(propulsion_id=ENGINE_WRAPPER), ivc)
    weight_b1_2 = problem.get_val("data:weight:propulsion:engine_oil:mass", units="kg")
    assert weight_b1_2 == pytest.approx(8.90, abs=1e-2)


def test_compute_engine_weight():
    """ Tests engine weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeEngineWeight(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEngineWeight(propulsion_id=ENGINE_WRAPPER), ivc)
    weight_b1 = problem.get_val("data:weight:propulsion:engine:mass", units="kg")
    assert weight_b1 == pytest.approx(357.41, abs=1e-2)


def test_compute_fuel_lines_weight():
    """ Tests fuel lines weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuelLinesWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelLinesWeight(), ivc)
    weight_b2 = problem.get_val("data:weight:propulsion:fuel_lines:mass", units="kg")
    assert weight_b2 == pytest.approx(57.3149, abs=1e-2)


def test_compute_unusable_fuel_weight():
    """ Tests engine weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnusableFuelWeight(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnusableFuelWeight(propulsion_id=ENGINE_WRAPPER), ivc)
    weight_b3 = problem.get_val("data:weight:propulsion:unusable_fuel:mass", units="kg")
    assert weight_b3 == pytest.approx(56.93, abs=1e-2)


def test_compute_navigation_systems_weight():
    """ Tests navigation systems weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeNavigationSystemsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNavigationSystemsWeight(), ivc)
    weight_c3 = problem.get_val("data:weight:systems:navigation:mass", units="kg")
    assert weight_c3 == pytest.approx(33.46, abs=1e-2)


def test_compute_power_systems_weight():
    """ Tests power systems weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePowerSystemsWeight()), __file__, XML_FILE)
    ivc.add_output("data:weight:systems:navigation:mass", 33.46, units="kg")
    ivc.add_output("data:weight:propulsion:fuel_lines:mass", 32.95, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePowerSystemsWeight(), ivc)
    weight_c12 = problem.get_val("data:weight:systems:power:electric_systems:mass", units="kg")
    assert weight_c12 == pytest.approx(72.53, abs=1e-2)
    weight_c13 = problem.get_val("data:weight:systems:power:hydraulic_systems:mass", units="kg")
    assert weight_c13 == pytest.approx(13.41, abs=1e-2)


def test_compute_life_support_systems_weight():
    """ Tests life support systems weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLifeSupportSystemsWeight()), __file__, XML_FILE)
    ivc.add_output("data:weight:systems:navigation:mass", 33.46, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLifeSupportSystemsWeight(), ivc)
    weight_c21 = problem.get_val("data:weight:systems:life_support:insulation:mass", units="kg")
    assert weight_c21 == pytest.approx(00., abs=1e-2)
    weight_c22 = problem.get_val("data:weight:systems:life_support:air_conditioning:mass", units="kg")
    assert weight_c22 == pytest.approx(44.717, abs=1e-2)
    weight_c23 = problem.get_val("data:weight:systems:life_support:de_icing:mass", units="kg")
    assert weight_c23 == pytest.approx(0., abs=1e-2)
    weight_c24 = problem.get_val("data:weight:systems:life_support:internal_lighting:mass", units="kg")
    assert weight_c24 == pytest.approx(0., abs=1e-2)
    weight_c25 = problem.get_val("data:weight:systems:life_support:seat_installation:mass", units="kg")
    assert weight_c25 == pytest.approx(0., abs=1e-2)
    weight_c26 = problem.get_val("data:weight:systems:life_support:fixed_oxygen:mass", units="kg")
    assert weight_c26 == pytest.approx(8.40, abs=1e-2)
    weight_c27 = problem.get_val("data:weight:systems:life_support:security_kits:mass", units="kg")
    assert weight_c27 == pytest.approx(0., abs=1e-2)


def test_compute_passenger_seats_weight():
    """ Tests passenger seats weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePassengerSeatsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePassengerSeatsWeight(), ivc)
    weight_d2 = problem.get_val("data:weight:furniture:passenger_seats:mass", units="kg")
    assert weight_d2 == pytest.approx(54.17, abs=1e-2)  # additional 2 pilots seats (differs from old version)


def test_evaluate_owe():
    """ Tests a simple evaluation of Operating Weight Empty from sample XML data. """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    # noinspection PyTypeChecker
    mass_computation = run_system(ComputeOperatingWeightEmpty(propulsion_id=ENGINE_WRAPPER), input_vars)

    oew = mass_computation.get_val("data:weight:aircraft:OWE", units="kg")
    assert oew == pytest.approx(1192.611, abs=1)


def test_loop_compute_owe():
    """ Tests a weight computation loop matching the max payload criterion. """

    # Payload is computed from NPAX_design
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read(
        ignore=[
            "data:weight:aircraft:max_payload",
            "data:weight:aircraft:MLW",
        ]
    ).to_ivc()
    input_vars.add_output("data:mission:sizing:fuel", 0.0, units="kg")

    # noinspection PyTypeChecker
    mass_computation = run_system(
        MassBreakdown(propulsion_id=ENGINE_WRAPPER, payload_from_npax=True),
        input_vars,
        check=True,
    )
    oew = mass_computation.get_val("data:weight:aircraft:OWE", units="kg")
    assert oew == pytest.approx(1199.571, abs=1)
