"""
test module for wing area computation.
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

import os.path as pth

import openmdao.api as om

from numpy.testing import assert_allclose

from ..wing_area_component.wing_area_loop_geom_simple import (
    UpdateWingAreaGeomSimple,
    ConstraintWingAreaGeomSimple,
)
from ..wing_area_component.wing_area_loop_cl_simple import (
    UpdateWingAreaLiftSimple,
    ConstraintWingAreaLiftSimple,
)
from ..wing_area_component.wing_area_loop_geom_adv import (
    UpdateWingAreaGeomAdvanced,
    ConstraintWingAreaGeomAdvanced,
)
from ..wing_area_component.wing_area_cl_equilibrium import (
    UpdateWingAreaLiftEquilibrium,
    ConstraintWingAreaLiftEquilibrium,
)
from ..wing_area_component.update_wing_area import UpdateWingArea
from ..update_wing_area_group import UpdateWingAreaGroup
from ..update_wing_position import UpdateWingPosition

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_update_wing_area_group():

    # Driven by fuel
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:fuel_type", 1.0)
    ivc.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc.add_output("data:mission:sizing:fuel", val=600.0, units="kg")
    ivc.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=573.00, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)

    problem = run_system(UpdateWingAreaGroup(), ivc)
    assert_allclose(problem["data:geometry:wing:area"], 20.05, atol=1e-2)
    assert_allclose(problem["data:constraints:wing:additional_CL_capacity"], 0.61, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"), -27, atol=1
    )
    # not 0.0 because MFW not updated

    # Driven by CL max
    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:fuel_type", 1.0)
    ivc.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc.add_output("data:mission:sizing:fuel", val=300.0, units="kg")
    ivc.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc.add_output("data:weight:aircraft:MFW", val=573.00, units="kg")
    ivc.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)

    problem = run_system(UpdateWingAreaGroup(), ivc)
    assert_allclose(problem["data:geometry:wing:area"], 14.02, atol=1e-2)
    assert_allclose(problem["data:constraints:wing:additional_CL_capacity"], 0.0, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"), 273, atol=1
    )


def test_simple_geom():

    ivc_loop = om.IndepVarComp()
    ivc_loop.add_output("data:propulsion:fuel_type", 1.0)
    ivc_loop.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc_loop.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc_loop.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc_loop.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc_loop.add_output("data:mission:sizing:fuel", val=600.0, units="kg")

    problem_loop = run_system(UpdateWingAreaGeomSimple(), ivc_loop)
    assert_allclose(problem_loop["wing_area"], 20.05, atol=1e-2)

    # _ = problem_loop.check_partials(compact_print=True)

    ivc_cons = om.IndepVarComp()
    ivc_cons.add_output("data:weight:aircraft:MFW", val=573.00, units="kg")
    ivc_cons.add_output("data:mission:sizing:fuel", val=600.0, units="kg")

    problem_cons = run_system(ConstraintWingAreaGeomSimple(), ivc_cons)
    assert_allclose(
        problem_cons.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"),
        -27.0,
        atol=1,
    )

    # _ = problem_cons.check_partials(compact_print=True)


def test_simple_cl():

    ivc_loop = om.IndepVarComp()
    ivc_loop.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc_loop.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc_loop.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)

    problem_loop = run_system(UpdateWingAreaLiftSimple(), ivc_loop)
    assert_allclose(problem_loop["wing_area"], 14.02, atol=1e-2)

    # _ = problem_loop.check_partials(compact_print=True)

    ivc_cons = om.IndepVarComp()
    ivc_cons.add_output("data:TLAR:v_approach", val=78.0, units="kn")
    ivc_cons.add_output("data:weight:aircraft:MLW", val=1692.37, units="kg")
    ivc_cons.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)
    ivc_cons.add_output("data:geometry:wing:area", val=14.02, units="m**2")

    problem_cons = run_system(ConstraintWingAreaLiftSimple(), ivc_cons)
    assert_allclose(
        problem_cons.get_val("data:constraints:wing:additional_CL_capacity"),
        0.0,
        atol=1e-2,
    )

    # _ = problem_cons.check_partials(compact_print=True)


def test_advanced_geom():

    ivc_loop = om.IndepVarComp()
    ivc_loop.add_output("data:propulsion:fuel_type", 1.0)
    ivc_loop.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc_loop.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc_loop.add_output("data:geometry:wing:kink:span_ratio", 0.0)
    ivc_loop.add_output("data:mission:sizing:fuel", val=600.0, units="kg")
    ivc_loop.add_output("data:aerodynamics:aircraft:landing:CL_max", val=2.0272)
    ivc_loop.add_output("data:geometry:wing:taper_ratio", val=0.8)
    ivc_loop.add_output("data:geometry:wing:aspect_ratio", val=4)
    ivc_loop.add_output("data:geometry:flap:chord_ratio", val=0.15)
    ivc_loop.add_output("data:geometry:wing:aileron:chord_ratio", val=0.2)
    ivc_loop.add_output("data:geometry:fuselage:maximum_width", val=1.5, units="m")
    ivc_loop.add_output("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=0.2)
    ivc_loop.add_output("data:geometry:propulsion:tank:y_ratio_tank_end", val=0.8)
    ivc_loop.add_output("data:geometry:propulsion:engine:layout", val=1.0)
    ivc_loop.add_output(
        "data:geometry:propulsion:engine:y_ratio",
        val=0.34,
    )
    ivc_loop.add_output("data:geometry:propulsion:tank:LE_chord_percentage", val=0.05)
    ivc_loop.add_output("data:geometry:propulsion:tank:TE_chord_percentage", val=0.05)
    ivc_loop.add_output("data:geometry:landing_gear:type", val=1.0)
    ivc_loop.add_output("data:geometry:landing_gear:y", val=1.5, units="m")
    ivc_loop.add_output("data:geometry:propulsion:nacelle:width", val=0.9291288709126333, units="m")
    ivc_loop.add_output("settings:geometry:fuel_tanks:depth", val=0.6)

    problem_loop = run_system(UpdateWingAreaGeomAdvanced(), ivc_loop)
    assert_allclose(problem_loop["wing_area"], 21.72, atol=1e-2)

    ivc_cons = om.IndepVarComp()
    ivc_cons.add_output("data:propulsion:fuel_type", 1.0)
    ivc_cons.add_output("data:geometry:wing:root:thickness_ratio", 0.149)
    ivc_cons.add_output("data:geometry:wing:tip:thickness_ratio", 0.103)
    ivc_cons.add_output("data:geometry:wing:kink:span_ratio", 0.0)
    ivc_cons.add_output("data:mission:sizing:fuel", val=600.0, units="kg")
    ivc_cons.add_output("data:geometry:wing:taper_ratio", val=0.8)
    ivc_cons.add_output("data:geometry:wing:aspect_ratio", val=4)
    ivc_cons.add_output("data:geometry:flap:chord_ratio", val=0.15)
    ivc_cons.add_output("data:geometry:wing:aileron:chord_ratio", val=0.2)
    ivc_cons.add_output("data:geometry:fuselage:maximum_width", val=1.5, units="m")
    ivc_cons.add_output("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=0.2)
    ivc_cons.add_output("data:geometry:propulsion:tank:y_ratio_tank_end", val=0.8)
    ivc_cons.add_output("data:geometry:propulsion:engine:layout", val=1.0)
    ivc_cons.add_output(
        "data:geometry:propulsion:engine:y_ratio",
        val=0.34,
    )
    ivc_cons.add_output("data:geometry:propulsion:tank:LE_chord_percentage", val=0.05)
    ivc_cons.add_output("data:geometry:propulsion:tank:TE_chord_percentage", val=0.05)
    ivc_cons.add_output("data:geometry:landing_gear:type", val=1.0)
    ivc_cons.add_output("data:geometry:landing_gear:y", val=1.5, units="m")
    ivc_cons.add_output("data:geometry:propulsion:nacelle:width", val=0.9291288709126333, units="m")
    ivc_cons.add_output("settings:geometry:fuel_tanks:depth", val=0.6)
    ivc_cons.add_output("data:geometry:wing:area", val=21.72, units="m**2")

    problem_cons = run_system(ConstraintWingAreaGeomAdvanced(), ivc_cons)
    assert_allclose(problem_cons["data:constraints:wing:additional_fuel_capacity"], 0.0, atol=1)


def test_advanced_cl():

    xml_file = "beechcraft_76.xml"

    inputs_list = list_inputs(
        UpdateWingAreaLiftEquilibrium(propulsion_id="fastga.wrapper.propulsion.basicIC_engine")
    )
    # Research independent input value in .xml file
    ivc_loop = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )
    ivc_loop.add_output("data:mission:sizing:taxi_in:thrust", val=1500, units="N")
    ivc_loop.add_output("data:mission:sizing:taxi_out:thrust", val=1500, units="N")

    problem_loop = run_system(
        UpdateWingAreaLiftEquilibrium(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 17.39, atol=1e-2)

    inputs_list = list_inputs(
        ConstraintWingAreaLiftEquilibrium(propulsion_id="fastga.wrapper.propulsion.basicIC_engine")
    )

    inputs_list.remove("data:geometry:wing:area")
    # Research independent input value in .xml file
    ivc_cons = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )
    ivc_cons.add_output("data:mission:sizing:taxi_in:thrust", val=1500, units="N")
    ivc_cons.add_output("data:mission:sizing:taxi_out:thrust", val=1500, units="N")
    ivc_cons.add_output("data:geometry:wing:area", val=17.39, units="m**2")
    problem_cons = run_system(
        ConstraintWingAreaLiftEquilibrium(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
        ivc_cons,
    )
    assert_allclose(
        problem_cons.get_val("data:constraints:wing:additional_CL_capacity"),
        0.0,
        atol=1e-2,
    )


def test_update_wing_area():

    ivc_geom = om.IndepVarComp()
    ivc_geom.add_output("wing_area:geometric", val=20.0, units="m**2")
    ivc_geom.add_output("wing_area:aerodynamic", val=15.0, units="m**2")

    problem_geom = run_system(UpdateWingArea(), ivc_geom)
    assert_allclose(problem_geom["data:geometry:wing:area"], 20.0, atol=1e-3)

    # _ = problem_geom.check_partials(compact_print=True)

    ivc_aero = om.IndepVarComp()
    ivc_aero.add_output("wing_area:geometric", val=10.0, units="m**2")
    ivc_aero.add_output("wing_area:aerodynamic", val=15.0, units="m**2")

    problem_aero = run_system(UpdateWingArea(), ivc_aero)
    assert_allclose(problem_aero["data:geometry:wing:area"], 15.0, atol=1e-3)

    # _ = problem_aero.check_partials(compact_print=True)


def test_update_wing_position():

    ivc = get_indep_var_comp(list_inputs(UpdateWingPosition()), __file__, "beechcraft_76.xml")

    problem = run_system(UpdateWingPosition(), ivc)
    assert_allclose(
        problem.get_val("data:geometry:wing:MAC:at25percent:x", units="m"), 3.4550, atol=1e-3
    )

    problem.check_partials(compact_print=True)
