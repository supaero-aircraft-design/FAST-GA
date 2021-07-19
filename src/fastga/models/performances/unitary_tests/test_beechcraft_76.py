"""
Test takeoff module
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

from openmdao.core.group import Group
import pytest

from ..takeoff import TakeOffPhase, _v2, _vr_from_v2, _v_lift_off_from_v2, _simulate_takeoff
from ..mission import _compute_taxi, _compute_climb, _compute_cruise, _compute_descent
from ..mission import Mission

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga.models.weight.cg.cg_variation import InFlightCGVariation

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

XML_FILE = "beechcraft_76.xml"


def test_v2():
    """ Tests safety speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    v2 = problem.get_val("v2:speed", units="m/s")
    assert v2 == pytest.approx(39.01, abs=1e-2)
    alpha = problem.get_val("v2:angle", units="deg")
    assert alpha == pytest.approx(8.23, abs=1e-2)
    climb_gradient = problem.get_val("v2:climb_gradient")
    assert climb_gradient == pytest.approx(0.22, abs=1e-2)


def test_vloff():
    """ Tests lift-off speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_v_lift_off_from_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("v2:speed", 39.01, units="m/s")
    ivc.add_output("v2:angle", 8.23, units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_v_lift_off_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vloff = problem.get_val("v_lift_off:speed", units="m/s")
    assert vloff == pytest.approx(38.10, abs=1e-2)
    alpha = problem.get_val("v_lift_off:angle", units="deg")
    assert alpha == pytest.approx(8.23, abs=1e-2)


def test_vr():
    """ Tests rotation speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_vr_from_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("v_lift_off:speed", 38.10, units="m/s")
    ivc.add_output("v_lift_off:angle", 8.23, units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_vr_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("vr:speed", units="m/s")
    assert vr == pytest.approx(29.91, abs=1e-2)


def test_simulate_takeoff():
    """ Tests simulate takeoff """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_simulate_takeoff(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("vr:speed", 29.91, units="m/s")
    ivc.add_output("v2:angle", 8.23, units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_simulate_takeoff(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units="m/s")
    assert vr == pytest.approx(38.16, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units="m/s")
    assert vloff == pytest.approx(42.74, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units="m/s")
    assert v2 == pytest.approx(44.99, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units="m")
    assert tofl == pytest.approx(500, abs=1)
    ground_roll = problem.get_val("data:mission:sizing:takeoff:ground_roll", units="m")
    assert ground_roll == pytest.approx(349, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units="s")
    assert duration == pytest.approx(19.3, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units="kg")
    assert fuel1 == pytest.approx(0.26, abs=1e-2)
    fuel2 = problem.get_val("data:mission:sizing:initial_climb:fuel", units="kg")
    assert fuel2 == pytest.approx(0.06, abs=1e-2)


def test_takeoff_phase_connections():
    """ Tests complete take-off phase connection with speeds """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(TakeOffPhase(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(TakeOffPhase(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units="m/s")
    assert vr == pytest.approx(38.16, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units="m/s")
    assert vloff == pytest.approx(42.73, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units="m/s")
    assert v2 == pytest.approx(44.99, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units="m")
    assert tofl == pytest.approx(500, abs=1)
    ground_roll = problem.get_val("data:mission:sizing:takeoff:ground_roll", units="m")
    assert ground_roll == pytest.approx(349, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units="s")
    assert duration == pytest.approx(19.3, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units="kg")
    assert fuel1 == pytest.approx(0.26, abs=1e-2)
    fuel2 = problem.get_val("data:mission:sizing:initial_climb:fuel", units="kg")
    assert fuel2 == pytest.approx(0.06, abs=1e-2)


def test_compute_taxi():
    """ Tests taxi in/out phase """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=True), ivc)
    fuel_mass = problem.get_val("data:mission:sizing:taxi_out:fuel", units="kg")
    assert fuel_mass == pytest.approx(
        0.23, abs=1e-2
    )  # result strongly dependent on the defined Thrust limit

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=False), ivc)
    fuel_mass = problem.get_val("data:mission:sizing:taxi_in:fuel", units="kg")
    assert fuel_mass == pytest.approx(
        0.23, abs=1e-2
    )  # result strongly dependent on the defined Thrust limit


def test_compute_climb():
    """ Tests climb phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("climb", _compute_climb(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    v_cas = problem.get_val("data:mission:sizing:main_route:climb:v_cas", units="kn")
    assert v_cas == pytest.approx(87.7, abs=1)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:climb:fuel", units="kg")
    assert fuel_mass == pytest.approx(4.23, abs=1e-2)
    distance = (
        problem.get_val("data:mission:sizing:main_route:climb:distance", units="m") / 1000.0
    )  # conversion to km
    assert distance == pytest.approx(24.55, abs=1e-2)
    duration = problem.get_val("data:mission:sizing:main_route:climb:duration", units="min")
    assert duration == pytest.approx(8.55, abs=1e-2)


def test_compute_cruise():
    """ Tests cruise phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("cruise", _compute_cruise(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:cruise:fuel", units="kg")
    assert fuel_mass == pytest.approx(116, abs=1)
    duration = problem.get_val("data:mission:sizing:main_route:cruise:duration", units="h")
    assert duration == pytest.approx(4.80, abs=1e-2)


def test_compute_descent():
    """ Tests descent phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("descent", _compute_descent(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:descent:fuel", units="kg")
    assert fuel_mass == pytest.approx(0.71, abs=1e-2)
    distance = (
        problem.get_val("data:mission:sizing:main_route:descent:distance", units="m") / 1000
    )  # conversion to km
    assert distance == pytest.approx(79, abs=1)
    duration = problem.get_val("data:mission:sizing:main_route:descent:duration", units="min")
    assert duration == pytest.approx(25, abs=1)


def test_loop_cruise_distance():
    """ Tests a distance computation loop matching the descent value/TLAR total range. """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Mission(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(Mission(propulsion_id=ENGINE_WRAPPER), ivc)
    m_total = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert m_total == pytest.approx(140, abs=1)
    climb_distance = problem.get_val("data:mission:sizing:main_route:climb:distance", units="NM")
    cruise_distance = problem.get_val("data:mission:sizing:main_route:cruise:distance", units="NM")
    descent_distance = problem.get_val(
        "data:mission:sizing:main_route:descent:distance", units="NM"
    )
    total_distance = problem.get_val("data:TLAR:range", units="NM")
    error_distance = total_distance - (climb_distance + cruise_distance + descent_distance)
    assert error_distance == pytest.approx(0.0, abs=1e-1)
