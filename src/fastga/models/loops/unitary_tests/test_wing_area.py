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

from ..update_wing_area import UpdateWingArea
from .... import models
from ...weight import mass_breakdown, cg
from fastga.command.api import generate_variables_description

from tests.testing_utilities import run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_update_wing_area():

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

    problem = run_system(UpdateWingArea(), ivc)
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

    problem = run_system(UpdateWingArea(), ivc)
    assert_allclose(problem["data:geometry:wing:area"], 14.02, atol=1e-2)
    assert_allclose(problem["data:constraints:wing:additional_CL_capacity"], 0.0, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"), 273, atol=1
    )


def test_write_variables():
    generate_variables_description(models.__path__[0], True)
