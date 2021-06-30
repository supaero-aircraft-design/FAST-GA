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

import os.path as pth
import os
import pandas as pd
import numpy as np
from openmdao.core.component import Component
from openmdao.core.group import Group
import pytest
from typing import Union

from fastoad.io import VariableIO
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.model_base.propulsion import IOMPropulsionWrapper

from ..takeoff import TakeOffPhase, _v2, _vr_from_v2, _vloff_from_v2, _simulate_takeoff
from ..mission import _compute_taxi, _compute_climb, _compute_cruise, _compute_descent
from ..mission import Mission

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga.models.weight.cg.cg_variation import InFlightCGVariation
from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

XML_FILE = "beechcraft_76.xml"
CSV_FILE = "flight_points.csv"


def test_v2():
    """ Tests safety speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    v2 = problem.get_val("v2:speed", units="m/s")
    assert v2 == pytest.approx(37.79, abs=1e-2)
    alpha = problem.get_val("v2:angle", units="deg")
    assert alpha == pytest.approx(8.49, abs=1e-2)


def test_vloff():
    """ Tests lift-off speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_vloff_from_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("v2:speed", 37.79, units="m/s")
    ivc.add_output("v2:angle", 8.49, units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_vloff_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vloff = problem.get_val("vloff:speed", units="m/s")
    assert vloff == pytest.approx(36.88, abs=1e-2)
    alpha = problem.get_val("vloff:angle", units="deg")
    assert alpha == pytest.approx(8.49, abs=1e-2)


def test_vr():
    """ Tests rotation speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_vr_from_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("vloff:speed", 36.88, units="m/s")
    ivc.add_output("vloff:angle", 8.49, units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_vr_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("vr:speed", units="m/s")
    assert vr == pytest.approx(28.53, abs=1e-2)


def test_simulate_takeoff():
    """ Tests simulate takeoff """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(_simulate_takeoff(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    ivc.add_output("vr:speed", 28.52, units="m/s")
    ivc.add_output("v2:angle", 8.49, units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_simulate_takeoff(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units="m/s")
    assert vr == pytest.approx(34.62, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units="m/s")
    assert vloff == pytest.approx(40.28, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units="m/s")
    assert v2 == pytest.approx(42.72, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units="m")
    assert tofl == pytest.approx(293, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units="s")
    assert duration == pytest.approx(17.5, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units="kg")
    assert fuel1 == pytest.approx(0.23, abs=1e-2)
    fuel2 = problem.get_val("data:mission:sizing:initial_climb:fuel", units="kg")
    assert fuel2 == pytest.approx(0.06, abs=1e-2)


def test_takeoff_phase_connections():
    """ Tests complete take-off phase connection with speeds """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read().to_ivc()
    # noinspection PyTypeChecker
    problem = run_system(TakeOffPhase(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units="m/s")
    assert vr == pytest.approx(34.62, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units="m/s")
    assert vloff == pytest.approx(40.28, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units="m/s")
    assert v2 == pytest.approx(42.72, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units="m")
    assert tofl == pytest.approx(293, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units="s")
    assert duration == pytest.approx(17.5, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units="kg")
    assert fuel1 == pytest.approx(0.23, abs=1e-2)
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
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.50, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    v_cas = problem.get_val("data:mission:sizing:main_route:climb:v_cas", units="kn")
    assert v_cas == pytest.approx(79.6, abs=1e-1)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:climb:fuel", units="kg")
    assert fuel_mass == pytest.approx(4.59, abs=1e-2)
    distance = (
        problem.get_val("data:mission:sizing:main_route:climb:distance", units="m") / 1000.0
    )  # conversion to km
    assert distance == pytest.approx(25.80, abs=1e-2)
    duration = problem.get_val("data:mission:sizing:main_route:climb:duration", units="min")
    assert duration == pytest.approx(9.94, abs=1e-2)


def test_compute_cruise():
    """ Tests cruise phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("cruise", _compute_cruise(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.50, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.56, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:distance", 13.2, units="km")
    ivc.add_output("data:mission:sizing:main_route:climb:duration", 9.94, units="min")
    ivc.add_output("data:mission:sizing:main_route:descent:distance", 0.0, units="km")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:cruise:fuel", units="kg")
    assert fuel_mass == pytest.approx(155.8, abs=1e-1)
    distance = (
        problem.get_val("data:mission:sizing:main_route:cruise:distance", units="m") / 1000.0
    )  # conversion to km
    assert distance == pytest.approx(1431.36, abs=1e-2)
    duration = problem.get_val("data:mission:sizing:main_route:cruise:duration", units="h")
    assert duration == pytest.approx(4.67, abs=1e-2)


def test_compute_descent():
    """ Tests descent phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("descent", _compute_descent(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.98, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.56, units="kg")
    ivc.add_output("data:mission:sizing:main_route:cruise:fuel", 188.05, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:distance", 25.80, units="km")
    ivc.add_output("data:mission:sizing:main_route:climb:duration", 9.94, units="min")
    ivc.add_output("data:mission:sizing:main_route:cruise:distance", 1431.36, units="km")
    ivc.add_output("data:mission:sizing:main_route:cruise:duration", 4.70, units="h")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:descent:fuel", units="kg")
    assert fuel_mass == pytest.approx(1.23, abs=1e-2)
    distance = (
        problem.get_val("data:mission:sizing:main_route:descent:distance", units="m") / 1000
    )  # conversion to km
    assert distance == pytest.approx(79.53, abs=1e-2)
    duration = problem.get_val("data:mission:sizing:main_route:descent:duration", units="min")
    assert duration == pytest.approx(25.13, abs=1e-2)


def test_loop_cruise_distance():
    """ Tests a distance computation loop matching the descent value/TLAR total range. """

    # Remove .csv file
    if pth.exists(pth.join(pth.split(__file__)[0], CSV_FILE)):
        os.remove(pth.join(pth.split(__file__)[0], CSV_FILE))

    # Get the parameters from .xml
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read().to_ivc()

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(Mission(propulsion_id=ENGINE_WRAPPER,
                                 out_file=pth.join(pth.split(__file__)[0], CSV_FILE)), ivc)
    m_total = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert m_total == pytest.approx(178.0, abs=1)
    climb_distance = problem.get_val("data:mission:sizing:main_route:climb:distance", units="NM")
    cruise_distance = problem.get_val("data:mission:sizing:main_route:cruise:distance", units="NM")
    descent_distance = problem.get_val(
        "data:mission:sizing:main_route:descent:distance", units="NM"
    )
    total_distance = problem.get_val("data:TLAR:range", units="NM")
    error_distance = total_distance - (climb_distance + cruise_distance + descent_distance)
    assert error_distance == pytest.approx(0.0, abs=1e-1)
    assert pth.exists(pth.join(pth.split(__file__)[0], CSV_FILE))

    # Remove .csv file
    if pth.exists(pth.join(pth.split(__file__)[0], CSV_FILE)):
        os.remove(pth.join(pth.split(__file__)[0], CSV_FILE))
