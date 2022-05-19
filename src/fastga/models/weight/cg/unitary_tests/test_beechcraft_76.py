"""Test module for geometry functions of cg components."""
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

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..cg import CG
from ..cg_components.a_airframe import (
    ComputeWingCG,
    ComputeFuselageCG,
    ComputeTailCG,
    ComputeFlightControlCG,
    ComputeLandingGearCG,
)
from ..cg_components.b_propulsion import (
    ComputeEngineCG,
    ComputeFuelLinesCG,
    ComputeTankCG,
    FuelPropulsionCG,
)
from ..cg_components.c_systems import (
    ComputePowerSystemsCG,
    ComputeLifeSupportCG,
    ComputeNavigationSystemsCG,
    ComputeRecordingSystemsCG,
)
from ..cg_components.d_furniture import ComputePassengerSeatsCG
from ..cg_components.payload import ComputePayloadCG
from ..cg_components.loadcase import ComputeGroundCGCase, ComputeFlightCGCase
from ..cg_components.ratio_aft import ComputeCGRatioAircraftEmpty
from ..cg_components.max_cg_ratio import ComputeMaxMinCGRatio

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

XML_FILE = "beechcraft_76.xml"


def test_compute_cg_wing():
    """Tests computation of wing center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingCG(), ivc)
    x_cg_a1 = problem.get_val("data:weight:airframe:wing:CG:x", units="m")
    assert x_cg_a1 == pytest.approx(3.71, abs=1e-2)


def test_compute_cg_fuselage():
    """Tests computation of fuselage center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageCG(), ivc)
    x_cg_a2 = problem.get_val("data:weight:airframe:fuselage:CG:x", units="m")
    assert x_cg_a2 == pytest.approx(3.50, abs=1e-2)


def test_compute_cg_tail():
    """Tests computation of tail center(s) of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeTailCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTailCG(), ivc)
    x_cg_a31 = problem.get_val("data:weight:airframe:horizontal_tail:CG:x", units="m")
    assert x_cg_a31 == pytest.approx(8.45, abs=1e-2)
    x_cg_a32 = problem.get_val("data:weight:airframe:vertical_tail:CG:x", units="m")
    assert x_cg_a32 == pytest.approx(7.9571, abs=1e-2)


def test_compute_cg_flight_control():
    """Tests computation of flight control center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFlightControlCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFlightControlCG(), ivc)
    x_cg_a4 = problem.get_val("data:weight:airframe:flight_controls:CG:x", units="m")
    assert x_cg_a4 == pytest.approx(5.72, abs=1e-2)


def test_compute_cg_landing_gear():
    """Tests computation of landing gear center(s) of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLandingGearCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLandingGearCG(), ivc)
    x_cg_a52 = problem.get_val("data:weight:airframe:landing_gear:front:CG:x", units="m")
    assert x_cg_a52 == pytest.approx(1.40, abs=1e-2)


def test_compute_cg_engine():
    """Tests computation of engine(s) center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeEngineCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEngineCG(), ivc)
    x_cg_b1 = problem.get_val("data:weight:propulsion:engine:CG:x", units="m")
    assert x_cg_b1 == pytest.approx(2.35946, abs=1e-2)


def test_compute_cg_fuel_lines():
    """Tests fuel lines center of gravity."""
    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs((ComputeFuelLinesCG())), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelLinesCG(), ivc)
    x_cg_b2 = problem.get_val("data:weight:propulsion:fuel_lines:CG:x", units="m")
    assert x_cg_b2 == pytest.approx(3.09027539, abs=1e-2)


def test_compute_cg_tank():
    """Tests tank center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeTankCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTankCG(), ivc)
    x_cg_b3 = problem.get_val("data:weight:propulsion:tank:CG:x", units="m")
    assert x_cg_b3 == pytest.approx(3.81, abs=1e-2)


def test_compute_cg_fuel_propulsion():
    """Tests whole fuel propulsion center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(FuelPropulsionCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FuelPropulsionCG(), ivc)
    x_cg_b3 = problem.get_val("data:weight:propulsion:CG:x", units="m")
    assert x_cg_b3 == pytest.approx(2.46, abs=1e-2)


def test_compute_cg_power_systems():
    """Tests computation of power systems center of gravity."""
    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePowerSystemsCG()), __file__, XML_FILE)
    ivc.add_output("data:weight:propulsion:engine:CG:x", 2.7, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePowerSystemsCG(), ivc)
    x_cg_c12 = problem.get_val("data:weight:systems:power:electric_systems:CG:x", units="m")
    assert x_cg_c12 == pytest.approx(3.32, abs=1e-2)
    x_cg_c13 = problem.get_val("data:weight:systems:power:hydraulic_systems:CG:x", units="m")
    assert x_cg_c13 == pytest.approx(3.32, abs=1e-2)


def test_compute_cg_life_support_systems():
    """Tests computation of life support systems center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLifeSupportCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLifeSupportCG(), ivc)
    x_cg_c22 = problem.get_val("data:weight:systems:life_support:air_conditioning:CG:x", units="m")
    assert x_cg_c22 == pytest.approx(1.87, abs=1e-2)


def test_compute_cg_navigation_systems():
    """Tests computation of navigation systems center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeNavigationSystemsCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNavigationSystemsCG(), ivc)
    x_cg_c3 = problem.get_val("data:weight:systems:avionics:CG:x", units="m")
    assert x_cg_c3 == pytest.approx(2.22, abs=1e-2)


def test_compute_cg_recording_systems():
    """Tests computation of navigation systems center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeRecordingSystemsCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeRecordingSystemsCG(), ivc)
    x_cg_c3 = problem.get_val("data:weight:systems:recording:CG:x", units="m")
    assert x_cg_c3 == pytest.approx(6.88, abs=1e-2)


def test_compute_cg_passenger_seats():
    """Tests computation of passenger seats center of gravity."""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePassengerSeatsCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePassengerSeatsCG(), ivc)
    x_cg_d2 = problem.get_val("data:weight:furniture:passenger_seats:CG:x", units="m")
    assert x_cg_d2 == pytest.approx(3.47, abs=1e-2)  # modified with new cabin definition


def test_compute_cg_payload():
    """Tests computation of payload center(s) of gravity."""
    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePayloadCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePayloadCG(), ivc)
    x_cg_pl = problem.get_val("data:weight:payload:PAX:CG:x", units="m")
    assert x_cg_pl == pytest.approx(3.47, abs=1e-1)
    x_cg_rear_fret = problem.get_val("data:weight:payload:rear_fret:CG:x", units="m")
    assert x_cg_rear_fret == pytest.approx(4.42, abs=1e-2)
    x_cg_front_fret = problem.get_val("data:weight:payload:front_fret:CG:x", units="m")
    assert x_cg_front_fret == pytest.approx(0.0, abs=1e-2)


def test_compute_cg_ratio_aft():
    """Tests computation of center of gravity with aft estimation."""
    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeCGRatioAircraftEmpty()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCGRatioAircraftEmpty(), ivc)
    empty_mass = problem.get_val("data:weight:aircraft_empty:mass", units="kg")
    assert empty_mass == pytest.approx(1109.06, abs=1e-2)
    cg_x = problem.get_val("data:weight:aircraft_empty:CG:x", units="m")
    assert cg_x == pytest.approx(3.22, abs=1e-2)
    cg_mac_pos = problem["data:weight:aircraft:empty:CG:MAC_position"]
    assert cg_mac_pos == pytest.approx(0.09, abs=1e-2)


def test_compute_cg_load_case():
    """Tests computation of center of gravity for ground/flight conf."""
    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeGroundCGCase()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeGroundCGCase(), ivc)
    mac_max = problem["data:weight:aircraft:CG:ground_condition:max:MAC_position"]
    assert mac_max == pytest.approx(0.287, abs=1e-3)
    mac_min = problem["data:weight:aircraft:CG:ground_condition:min:MAC_position"]
    assert mac_min == pytest.approx(0.198, abs=1e-2)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(
        list_inputs(ComputeFlightCGCase(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFlightCGCase(propulsion_id=ENGINE_WRAPPER), ivc)
    mac_max = problem["data:weight:aircraft:CG:flight_condition:max:MAC_position"]
    assert mac_max == pytest.approx(0.321, abs=1e-3)
    mac_min = problem["data:weight:aircraft:CG:flight_condition:min:MAC_position"]
    assert mac_min == pytest.approx(0.198, abs=1e-2)


def test_compute_max_cg_ratio():
    """Tests computation of maximum center of gravity ratio."""
    # Define the independent input values that should be filled if basic function is chosen
    ivc = get_indep_var_comp(list_inputs(ComputeMaxMinCGRatio()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMaxMinCGRatio(), ivc)
    cg_ratio_aft = problem.get_val("data:weight:aircraft:CG:aft:MAC_position")
    assert cg_ratio_aft == pytest.approx(0.369, abs=1e-3)
    cg_ratio_fwd = problem.get_val("data:weight:aircraft:CG:fwd:MAC_position")
    assert cg_ratio_fwd == pytest.approx(0.152, abs=1e-3)


def test_complete_cg():
    """Run computation of all models."""
    # with data from file
    ivc = get_indep_var_comp(list_inputs(CG(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(CG(propulsion_id=ENGINE_WRAPPER), ivc, check=True)
    cg_global = problem.get_val("data:weight:aircraft:CG:aft:x", units="m")
    assert cg_global == pytest.approx(3.527, abs=1e-3)
    cg_ratio = problem.get_val("data:weight:aircraft:CG:aft:MAC_position")
    assert cg_ratio == pytest.approx(0.299, abs=1e-3)
    z_cg_empty_ac = problem.get_val("data:weight:aircraft_empty:CG:z", units="m")
    assert z_cg_empty_ac == pytest.approx(1.21, abs=1e-3)
    z_cg_b1 = problem.get_val("data:weight:propulsion:engine:CG:z", units="m")
    assert z_cg_b1 == pytest.approx(1.195, abs=1e-2)
