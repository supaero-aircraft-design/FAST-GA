"""
test module for wing area computation
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

import openmdao.api as om

from numpy.testing import assert_allclose

from ..update_wing_area_simple import UpdateWingAreaSimple
from ..update_wing_area_advanced import UpdateWingAreaAdvanced

from tests.testing_utilities import run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_update_wing_area_simple():

    # Driven by fuel
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc.add_output("data:mission:sizing:fuel", val=600.0, units="kg")
    ivc.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=573.00, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)

    problem = run_system(UpdateWingAreaSimple(), ivc)
    assert_allclose(problem["data:geometry:wing:area"], 20.05, atol=1e-2)
    assert_allclose(problem["data:constraints:wing:additional_CL_capacity"], 0.61, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"), -27, atol=1
    )
    # not 0.0 because MFW not updated

    # Driven by CL max
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc.add_output("data:mission:sizing:fuel", val=300.0, units="kg")
    ivc.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=573.00, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)

    problem = run_system(UpdateWingAreaSimple(), ivc)
    assert_allclose(problem["data:geometry:wing:area"], 14.02, atol=1e-2)
    assert_allclose(problem["data:constraints:wing:additional_CL_capacity"], 0.0, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"), 273, atol=1
    )


def test_update_wing_area_advanced():
    # Driven by fuel
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc.add_output("data:geometry:wing:kink:span_ratio", 0.0)
    ivc.add_output("data:mission:sizing:fuel", val=600.0, units="kg")
    ivc.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=600.00, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)
    ivc.add_output("data:geometry:wing:taper_ratio", val=0.8)
    ivc.add_output("data:geometry:wing:aspect_ratio", val=4)
    ivc.add_output("data:geometry:flap:chord_ratio", val=0.15)
    ivc.add_output("data:geometry:aileron:chord_ratio", val=0.2)
    ivc.add_output("data:geometry:fuselage:maximum_width", val=1.5, units="m")
    ivc.add_output("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=0.2)
    ivc.add_output("data:geometry:propulsion:tank:y_ratio_tank_end", val=0.8)
    ivc.add_output("data:geometry:propulsion:engine:layout", val=1.0)
    ivc.add_output(
        "data:geometry:propulsion:engine:y_ratio",
        shape=10,
        val=[0.34, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    )
    ivc.add_output("data:geometry:propulsion:tank:LE_chord_percentage", val=0.05)
    ivc.add_output("data:geometry:propulsion:tank:TE_chord_percentage", val=0.05)
    ivc.add_output("data:geometry:landing_gear:type", val=1.0)
    ivc.add_output("data:geometry:landing_gear:y", val=1.5, units="m")
    ivc.add_output("data:geometry:propulsion:nacelle:width", val=0.9291288709126333, units="m")
    ivc.add_output("settings:geometry:fuel_tanks:depth", val=0.6)

    problem = run_system(UpdateWingAreaAdvanced(), ivc)
    assert_allclose(problem["data:geometry:wing:area"], 21.72, atol=1e-2)
    assert_allclose(problem["data:constraints:wing:additional_CL_capacity"], 0.721, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"), 0.0, atol=1
    )
    # not 0.0 because MFW not updated

    # Driven by CL max
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc.add_output("data:geometry:wing:kink:span_ratio", 0.0)
    ivc.add_output("data:mission:sizing:fuel", val=250.0, units="kg")
    ivc.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=500.00, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)
    ivc.add_output("data:geometry:wing:taper_ratio", val=0.8)
    ivc.add_output("data:geometry:wing:aspect_ratio", val=4)
    ivc.add_output("data:geometry:flap:chord_ratio", val=0.15)
    ivc.add_output("data:geometry:aileron:chord_ratio", val=0.2)
    ivc.add_output("data:geometry:fuselage:maximum_width", val=1.5, units="m")
    ivc.add_output("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=0.2)
    ivc.add_output("data:geometry:propulsion:tank:y_ratio_tank_end", val=0.8)
    ivc.add_output("data:geometry:propulsion:engine:layout", val=1.0)
    ivc.add_output(
        "data:geometry:propulsion:engine:y_ratio",
        shape=10,
        val=[0.34, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    )
    ivc.add_output("data:geometry:propulsion:tank:LE_chord_percentage", val=0.05)
    ivc.add_output("data:geometry:propulsion:tank:TE_chord_percentage", val=0.05)
    ivc.add_output("data:geometry:landing_gear:type", val=1.0)
    ivc.add_output("data:geometry:landing_gear:y", val=1.5, units="m")
    ivc.add_output("data:geometry:propulsion:nacelle:width", val=0.9291288709126333, units="m")
    ivc.add_output("settings:geometry:fuel_tanks:depth", val=0.6)

    problem = run_system(UpdateWingAreaAdvanced(), ivc)
    assert_allclose(problem["data:geometry:wing:area"], 14.02, atol=1e-2)
    assert_allclose(problem["data:constraints:wing:additional_CL_capacity"], 0.0, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"), 250, atol=1
    )
