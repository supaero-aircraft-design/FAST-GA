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

import pytest

from ..a_airframe import (
    ComputeTailWeight,
    ComputeFlightControlsWeight,
    ComputeFuselageWeight,
    ComputeFuselageWeightAlternate,
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
    ComputeNavigationSystemsWeightFLOPS,
)
from ..d_furniture import ComputePassengerSeatsWeight
from ..mass_breakdown import MassBreakdown, ComputeOperatingWeightEmpty
from ..payload import ComputePayload

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER

XML_FILE = "cirrus_sr22.xml"


def test_compute_payload():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePayload()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePayload(), ivc)
    assert problem["data:weight:aircraft:payload"] == pytest.approx(355.0, abs=1e-2)
    assert problem["data:weight:aircraft:max_payload"] == pytest.approx(420.0, abs=1e-2)


def test_compute_wing_weight():
    """ Tests wing weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingWeight(), ivc)
    weight_a1 = problem.get_val("data:weight:airframe:wing:mass", units="kg")
    assert weight_a1 == pytest.approx(
        162.86, abs=1e-2
    )  # difference because of integer conversion error


def test_compute_fuselage_weight():
    """ Tests fuselage weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeight(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass", units="kg")
    assert weight_a2 == pytest.approx(140.39, abs=1e-2)


def test_compute_fuselage_weight_alternate():
    """ Tests alternate fuselage weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageWeightAlternate(propulsion_id=ENGINE_WRAPPER)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("data:flight_domain:diving_speed", 109.139)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeightAlternate(propulsion_id=ENGINE_WRAPPER), ivc)
    skin_thickness = problem.get_val("data:geometry:fuselage:skin_thickness", units="m")
    assert skin_thickness == pytest.approx(0.0005, abs=1e-4)
    shell_mass = problem.get_val("data:weight:airframe:fuselage:shell_mass", units="kg")
    assert shell_mass == pytest.approx(48.721, abs=1e-3)
    cone_mass = problem.get_val("data:weight:airframe:fuselage:tail_cone_mass", units="kg")
    assert cone_mass == pytest.approx(12.656, abs=1e-3)
    nose_mass = problem.get_val("data:weight:airframe:fuselage:nose_mass", units="kg")
    assert nose_mass == pytest.approx(9.305, abs=1e-3)
    pax_windows_mass = problem.get_val("data:weight:airframe:fuselage:pax_windows_mass", units="kg")
    assert pax_windows_mass == pytest.approx(9.98, abs=1e-3)
    cockpit_window_mass = problem.get_val(
        "data:weight:airframe:fuselage:cockpit_window_mass", units="kg"
    )
    assert cockpit_window_mass == pytest.approx(6.765, abs=1e-3)
    nlg_door_mass = problem.get_val("data:weight:airframe:fuselage:nlg_door_mass", units="kg")
    assert nlg_door_mass == pytest.approx(8.878, abs=1e-3)
    doors_mass = problem.get_val("data:weight:airframe:fuselage:doors_mass", units="kg")
    assert doors_mass == pytest.approx(35.135, abs=1e-3)
    wing_fuselage_connection_mass = problem.get_val(
        "data:weight:airframe:fuselage:wing_fuselage_connection_mass", units="kg"
    )
    assert wing_fuselage_connection_mass == pytest.approx(19.907, abs=1e-3)
    engine_support_mass = problem.get_val(
        "data:weight:airframe:fuselage:engine_support_mass", units="kg"
    )
    assert engine_support_mass == pytest.approx(18.2, abs=1e-3)
    floor_mass = problem.get_val("data:weight:airframe:fuselage:floor_mass", units="kg")
    assert floor_mass == pytest.approx(17.825, abs=1e-3)
    bulkheads_mass = problem.get_val("data:weight:airframe:fuselage:bulkheads_mass", units="kg")
    assert bulkheads_mass == pytest.approx(0, abs=1e-3)
    fuselage_mass = problem.get_val("data:weight:airframe:fuselage:mass", units="kg")
    assert fuselage_mass == pytest.approx(165.413, abs=1e-3)
    total_additional_mass = problem.get_val(
        "data:weight:airframe:fuselage:total_additional_mass", units="kg"
    )
    assert total_additional_mass == pytest.approx(116.692, abs=1e-3)
    fuselage_inertia = problem.get_val("data:loads:fuselage:inertia", units="m**4")
    assert fuselage_inertia == pytest.approx(0.000499, abs=1e-5)
    sigma_mh = problem.get_val("data:loads:fuselage:sigmaMh", units="N/m**2")
    assert sigma_mh == pytest.approx(1.8e08, abs=1e-3)


def test_compute_fuselage_weight_raymer():
    """ Tests fuselage weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageWeightRaymer()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeightRaymer(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass_raymer", units="kg")
    assert weight_a2 == pytest.approx(118.92, abs=1e-2)


def test_compute_empennage_weight():
    """ Tests empennage weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeTailWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTailWeight(), ivc)
    weight_a31 = problem.get_val("data:weight:airframe:horizontal_tail:mass", units="kg")
    assert weight_a31 == pytest.approx(14.96, abs=1e-2)
    weight_a32 = problem.get_val("data:weight:airframe:vertical_tail:mass", units="kg")
    assert weight_a32 == pytest.approx(10.04, abs=1e-2)


def test_compute_flight_controls_weight():
    """ Tests flight controls weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFlightControlsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFlightControlsWeight(), ivc)
    weight_a4 = problem.get_val("data:weight:airframe:flight_controls:mass", units="kg")
    assert weight_a4 == pytest.approx(22.32, abs=1e-2)


def test_compute_landing_gear_weight():
    """ Tests landing gear weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLandingGearWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLandingGearWeight(), ivc)
    weight_a51 = problem.get_val("data:weight:airframe:landing_gear:main:mass", units="kg")
    assert weight_a51 == pytest.approx(39.68, abs=1e-2)
    weight_a52 = problem.get_val("data:weight:airframe:landing_gear:front:mass", units="kg")
    assert weight_a52 == pytest.approx(16.49, abs=1e-2)


def test_compute_oil_weight():
    """ Tests engine weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeOilWeight(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOilWeight(propulsion_id=ENGINE_WRAPPER), ivc)
    weight_b1_2 = problem.get_val("data:weight:propulsion:engine_oil:mass", units="kg")
    assert weight_b1_2 == pytest.approx(2.836, abs=1e-2)


def test_compute_engine_weight():
    """ Tests engine weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeEngineWeight(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEngineWeight(propulsion_id=ENGINE_WRAPPER), ivc)
    weight_b1 = problem.get_val("data:weight:propulsion:engine:mass", units="kg")
    assert weight_b1 == pytest.approx(330.21, abs=1e-2)


def test_compute_fuel_lines_weight():
    """ Tests fuel lines weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuelLinesWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelLinesWeight(), ivc)
    weight_b2 = problem.get_val("data:weight:propulsion:fuel_lines:mass", units="kg")
    assert weight_b2 == pytest.approx(31.30, abs=1e-2)


def test_compute_unusable_fuel_weight():
    """ Tests engine weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeUnusableFuelWeight(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnusableFuelWeight(propulsion_id=ENGINE_WRAPPER), ivc)
    weight_b3 = problem.get_val("data:weight:propulsion:unusable_fuel:mass", units="kg")
    assert weight_b3 == pytest.approx(33.25, abs=1e-2)


def test_compute_navigation_systems_weight():
    """ Tests navigation systems weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeNavigationSystemsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNavigationSystemsWeight(), ivc)
    weight_c3 = problem.get_val("data:weight:systems:navigation:mass", units="kg")
    assert weight_c3 == pytest.approx(59.874, abs=1e-2)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeNavigationSystemsWeightFLOPS()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNavigationSystemsWeightFLOPS(), ivc)
    weight_c3 = problem.get_val("data:weight:systems:navigation:mass", units="kg")
    assert weight_c3 == pytest.approx(2896.16, abs=1e-2)
    weight_c31 = problem.get_val("data:weight:systems:navigation:instruments:mass", units="kg")
    assert weight_c31 == pytest.approx(28.24, abs=1e-2)
    weight_c32 = problem.get_val("data:weight:systems:navigation:avionics:mass", units="kg")
    assert weight_c32 == pytest.approx(2867.91, abs=1e-2)


def test_compute_power_systems_weight():
    """ Tests power systems weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePowerSystemsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePowerSystemsWeight(), ivc)
    weight_c12 = problem.get_val("data:weight:systems:power:electric_systems:mass", units="kg")
    assert weight_c12 == pytest.approx(85.25, abs=1e-2)
    weight_c13 = problem.get_val("data:weight:systems:power:hydraulic_systems:mass", units="kg")
    assert weight_c13 == pytest.approx(11.29, abs=1e-2)


def test_compute_life_support_systems_weight():
    """ Tests life support systems weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLifeSupportSystemsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLifeSupportSystemsWeight(), ivc)
    weight_c21 = problem.get_val("data:weight:systems:life_support:insulation:mass", units="kg")
    assert weight_c21 == pytest.approx(00.0, abs=1e-2)
    weight_c22 = problem.get_val(
        "data:weight:systems:life_support:air_conditioning:mass", units="kg"
    )
    assert weight_c22 == pytest.approx(45.31, abs=1e-2)
    weight_c23 = problem.get_val("data:weight:systems:life_support:de_icing:mass", units="kg")
    assert weight_c23 == pytest.approx(0.0, abs=1e-2)
    weight_c24 = problem.get_val(
        "data:weight:systems:life_support:internal_lighting:mass", units="kg"
    )
    assert weight_c24 == pytest.approx(0.0, abs=1e-2)
    weight_c25 = problem.get_val(
        "data:weight:systems:life_support:seat_installation:mass", units="kg"
    )
    assert weight_c25 == pytest.approx(0.0, abs=1e-2)
    weight_c26 = problem.get_val("data:weight:systems:life_support:fixed_oxygen:mass", units="kg")
    assert weight_c26 == pytest.approx(8.40, abs=1e-2)
    weight_c27 = problem.get_val("data:weight:systems:life_support:security_kits:mass", units="kg")
    assert weight_c27 == pytest.approx(0.0, abs=1e-2)


def test_compute_passenger_seats_weight():
    """ Tests passenger seats weight computation from sample XML data """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePassengerSeatsWeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePassengerSeatsWeight(), ivc)
    weight_d2 = problem.get_val("data:weight:furniture:passenger_seats:mass", units="kg")
    assert weight_d2 == pytest.approx(
        49.82, abs=1e-2
    )  # additional 2 pilots seats (differs from old version)


def test_evaluate_owe():
    """ Tests a simple evaluation of Operating Weight Empty from sample XML data. """

    ivc = get_indep_var_comp(
        list_inputs(ComputeOperatingWeightEmpty(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("data:flight_domain:diving_speed", 109.139)

    # noinspection PyTypeChecker
    mass_computation = run_system(ComputeOperatingWeightEmpty(propulsion_id=ENGINE_WRAPPER), ivc)

    oew = mass_computation.get_val("data:weight:aircraft:OWE", units="kg")
    assert oew == pytest.approx(1053, abs=1)


def test_loop_compute_owe():
    """ Tests a weight computation loop matching the max payload criterion. """

    # Payload is computed from NPAX_design
    ivc = get_indep_var_comp(
        list_inputs(MassBreakdown(propulsion_id=ENGINE_WRAPPER, payload_from_npax=True)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("data:flight_domain:diving_speed", 109.139)

    # noinspection PyTypeChecker
    mass_computation = run_system(
        MassBreakdown(propulsion_id=ENGINE_WRAPPER, payload_from_npax=True), ivc, check=True,
    )
    oew = mass_computation.get_val("data:weight:aircraft:OWE", units="kg")
    assert oew == pytest.approx(1053, abs=1)
