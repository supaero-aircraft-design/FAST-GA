"""
Computation of wing area update and constraints based on the lift required in low speed
conditions with an equilibrium computation.
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

import logging

import numpy as np
import openmdao.api as om

from scipy.constants import g

import fastoad.api as oad
from fastoad.openmdao.problem import AutoUnitsDefaultGroup
from fastoad.constants import EngineSetting

from fastga.command.api import list_inputs_metadata

from fastga.models.performances.mission_vector.mission.dep_equilibrium import (
    DEPEquilibrium,
)
from fastga.models.performances.mission_vector.constants import SUBMODEL_EQUILIBRIUM

from ..constants import SUBMODEL_WING_AREA_AERO_LOOP, SUBMODEL_WING_AREA_AERO_CONS

_LOGGER = logging.getLogger(__name__)


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_AERO_LOOP, "fastga.submodel.loop.wing_area.update.aero.equilibrium"
)
class UpdateWingAreaLiftEquilibrium(om.ExplicitComponent):
    """
    Computes needed wing area to reach an equilibrium at required approach speed.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:x", val=np.nan, units="m")

        input_zip = zip_equilibrium_input(self.options["propulsion_id"])
        for var_names, var_unit, var_shape, var_shape_by_conn, var_copy_shape in input_zip:
            if var_names[:5] == "data:" and var_names != "data:geometry:wing:area":
                if var_shape_by_conn:
                    self.add_input(
                        name=var_names,
                        val=np.nan,
                        units=var_unit,
                        shape_by_conn=var_shape_by_conn,
                        copy_shape=var_copy_shape,
                    )
                else:
                    self.add_input(
                        name=var_names,
                        val=np.nan,
                        units=var_unit,
                        shape=var_shape,
                    )

        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:mission:sizing:landing:elevator_angle", val=np.nan, units="deg")
        self.add_input("data:mission:sizing:takeoff:elevator_angle", val=np.nan, units="deg")

        self.add_output("wing_area", val=10.0, units="m**2")

        self.declare_partials(
            "wing_area",
            "*",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # First, compute a failsafe value, in case the computation crashes because of the wrong
        # initial guesses of the problem
        stall_speed = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]

        wing_area_landing_init_guess = 2 * mlw * g / (stall_speed ** 2) / (1.225 * max_cl)

        wing_area_approach = compute_wing_area(inputs, self.options["propulsion_id"])

        if wing_area_approach > 1.2 * wing_area_landing_init_guess:
            print("Vrai valeur", wing_area_approach)
            wing_area_approach = wing_area_landing_init_guess
            _LOGGER.info(
                "Wing area too far from potential data, taking backup value for this iteration"
            )

        outputs["wing_area"] = wing_area_approach


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_AERO_CONS, "fastga.submodel.loop.wing_area.constraint.aero.equilibrium"
)
class ConstraintWingAreaLiftEquilibrium(om.ExplicitComponent):
    """
    Computes the difference between the lift coefficient required for the low speed conditions
    and the what the wing can provide while maintaining an equilibrium. Will be an equivalent
    lift coefficient since the maximum one cannot be computed so easily. Equivalence will be
    computed based on the lift equation.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:x", val=np.nan, units="m")

        input_zip = zip_equilibrium_input(self.options["propulsion_id"])
        for var_names, var_unit, var_shape, var_shape_by_conn, var_copy_shape in input_zip:
            if var_names[:5] == "data:" and var_names != "data:geometry:wing:area":
                if var_shape_by_conn:
                    self.add_input(
                        name=var_names,
                        val=np.nan,
                        units=var_unit,
                        shape_by_conn=var_shape_by_conn,
                        copy_shape=var_copy_shape,
                    )
                else:
                    self.add_input(
                        name=var_names,
                        val=np.nan,
                        units=var_unit,
                        shape=var_shape,
                    )

        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:mission:sizing:landing:elevator_angle", val=np.nan, units="deg")
        self.add_input("data:mission:sizing:takeoff:elevator_angle", val=np.nan, units="deg")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:constraints:wing:additional_CL_capacity")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        v_stall = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area_actual = inputs["data:geometry:wing:area"]

        wing_area_constraint = compute_wing_area(inputs, self.options["propulsion_id"])

        additional_cl = (
            (2.0 * mlw * g)
            / (1.225 * v_stall ** 2.0)
            * (1.0 / wing_area_constraint - 1.0 / wing_area_actual)
        )

        outputs["data:constraints:wing:additional_CL_capacity"] = additional_cl


class _IDThrustRate(om.ExplicitComponent):
    def setup(self):
        self.add_input("thrust_rate_t_econ", shape=4, val=np.full(4, np.nan))
        self.add_output("thrust_rate", shape=2)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["thrust_rate"] = inputs["thrust_rate_t_econ"][0:2]


def compute_wing_area(inputs, propulsion_id):

    # First, setup an initial guess
    stall_speed = inputs["data:TLAR:v_approach"] / 1.3
    mlw = inputs["data:weight:aircraft:MLW"]
    cg_max_aft = float(inputs["data:weight:aircraft:CG:aft:x"])
    cg_max_fwd = float(inputs["data:weight:aircraft:CG:fwd:x"])
    delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
    cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
    cl_0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
    max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]
    min_elevator_angle = min(
        inputs["data:mission:sizing:landing:elevator_angle"],
        inputs["data:mission:sizing:takeoff:elevator_angle"],
    )
    wing_area_landing_init_guess = 2 * mlw * g / (stall_speed ** 2) / (1.225 * max_cl)

    alpha_max = (max_cl - delta_cl_flaps - cl_0_wing) / cl_alpha * 180.0 / np.pi

    input_zip = zip_equilibrium_input(propulsion_id)

    ivc = om.IndepVarComp()
    for var_names, var_unit, _, _, _ in input_zip:
        if var_names[:5] == "data:" and var_names != "data:geometry:wing:area":
            ivc.add_output(
                name=var_names,
                val=inputs[var_names],
                units=var_unit,
                shape=np.shape(inputs[var_names]),
            )

    ivc.add_output(name="d_vx_dt", val=np.array([0.0, 0.0]), units="m/s**2")
    ivc.add_output(name="mass", val=np.array([mlw, mlw]), units="kg")
    # x_cg should be evaluated at the worst case scenario so either max aft or max fwd
    ivc.add_output(name="x_cg", val=np.array([cg_max_fwd, cg_max_aft]), units="m")
    ivc.add_output(name="gamma", val=np.array([0.0, 0.0]), units=None)
    ivc.add_output(name="altitude", val=np.array([0.0, 0.0]), units="m")
    # Time step is not important since we don't care about the fuel consumption
    ivc.add_output(name="time_step", val=np.array([0.0, 0.0]), units="s")
    ivc.add_output(name="true_airspeed", val=np.array([stall_speed, stall_speed]), units="m/s")
    ivc.add_output(name="engine_setting", val=np.full(2, EngineSetting.TAKEOFF))

    problem = om.Problem()
    model = problem.model

    model.add_subsystem("ivc", ivc, promotes_outputs=["*"])
    model.add_subsystem(
        "Equilibrium",
        DEPEquilibrium(
            number_of_points=2,
            promotes_all_variables=True,
            propulsion_id=propulsion_id,
            flaps_position="landing",
        ),
        promotes=["*"],
    )
    model.add_subsystem("thrust_rate_id", _IDThrustRate(), promotes=["*"])

    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.nonlinear_solver.options["iprint"] = 2
    model.nonlinear_solver.options["maxiter"] = 100
    model.nonlinear_solver.options["rtol"] = 1e-4
    model.linear_solver = om.DirectSolver()

    problem.driver = om.ScipyOptimizeDriver()
    problem.driver.options["optimizer"] = "SLSQP"
    problem.driver.options["maxiter"] = 100
    problem.driver.options["tol"] = 1e-4

    problem.model.add_design_var(
        name="data:geometry:wing:area",
        units="m**2",
        lower=1.0,
        upper=2.0 * wing_area_landing_init_guess,
    )

    problem.model.add_objective(name="data:geometry:wing:area", units="m**2")

    problem.model.add_constraint(
        name="alpha", units="deg", lower=[0.0, 0.0], upper=[alpha_max, alpha_max]
    )
    problem.model.add_constraint(name="thrust_rate", lower=[0.0, 0.0], upper=[1.0, 1.0])
    problem.model.add_constraint(
        name="delta_m",
        lower=[min_elevator_angle, min_elevator_angle],
        upper=[abs(min_elevator_angle), abs(min_elevator_angle)],
    )

    problem.model.approx_totals()

    problem.setup()

    problem["data:geometry:wing:area"] = wing_area_landing_init_guess
    problem["delta_m"] = np.array([0.9 * min_elevator_angle, 0.9 * min_elevator_angle])
    problem["alpha"] = np.array([0.9 * alpha_max, 0.9 * alpha_max])

    problem.run_driver()

    print(problem["delta_m"])
    print(problem["alpha"])

    wing_area_approach = problem.get_val("data:geometry:wing:area", units="m**2")

    return wing_area_approach


def zip_equilibrium_input(propulsion_id):
    """
    Returns a list of the variables needed for the computation of the equilibrium. Based on
    the submodel currently registered and the propulsion_id required.

    :param propulsion_id: ID of propulsion wrapped to be used for computation of equilibrium.
    :return inputs_zip: a zip containing a list of name, a list of units, a list of shapes,
    a list of shape_by_conn boolean and a list of copy_shape str.
    """
    new_component = AutoUnitsDefaultGroup()
    option_equilibrium = {
        "number_of_points": 2,
        "promotes_all_variables": True,
        "propulsion_id": propulsion_id,
        "flaps_position": "landing",
    }
    new_component.add_subsystem(
        "system",
        oad.RegisterSubmodel.get_submodel(SUBMODEL_EQUILIBRIUM, options=option_equilibrium),
        promotes=["*"],
    )

    name, unit, shape, shape_by_conn, copy_shape = list_inputs_metadata(new_component)
    inputs_zip = zip(name, unit, shape, shape_by_conn, copy_shape)

    return inputs_zip
