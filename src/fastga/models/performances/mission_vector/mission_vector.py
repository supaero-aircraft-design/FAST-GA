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

import openmdao.api as om

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
    POINTS_NB_DESCENT,
)
from fastga.models.performances.mission_vector.initialization.initialize import Initialize
from fastga.models.performances.mission_vector.mission.mission_core import MissionCore
from fastga.models.performances.mission_vector.to_csv import ToCSV
from fastga.models.weight.cg.cg_variation import InFlightCGVariation


class MissionVector(om.Group):
    """Computes and potentially save mission based on options."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.DirectSolver()

    def initialize(self):

        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)
        self.options.declare("out_file", default="", types=str)

    def setup(self):

        number_of_points = POINTS_NB_CLIMB + POINTS_NB_CRUISE + POINTS_NB_DESCENT

        self.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
        self.add_subsystem(
            "initialization",
            Initialize(number_of_points=number_of_points),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "solve_equilibrium",
            MissionCore(
                number_of_points=number_of_points, propulsion_id=self.options["propulsion_id"]
            ),
            promotes_inputs=["data:*", "settings:*"],
            promotes_outputs=["data:*"],
        )
        self.add_subsystem(
            "to_csv",
            ToCSV(number_of_points=number_of_points, out_file=self.options["out_file"]),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )

        self.connect(
            "initialization.initialize_engine_setting.engine_setting",
            [
                "solve_equilibrium.compute_dep_equilibrium.preparation_for_energy_consumption"
                + ".engine_setting",
                "to_csv.engine_setting",
            ],
        )

        self.connect(
            "initialization.initialize_center_of_gravity.x_cg",
            [
                "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.x_cg",
                "to_csv.x_cg",
            ],
        )

        self.connect(
            "initialization.initialize_time_and_distance.position",
            ["solve_equilibrium.performance_per_phase.position", "to_csv.position"],
        )

        self.connect(
            "initialization.initialize_time_and_distance.time",
            [
                "solve_equilibrium.compute_time_step.time",
                "solve_equilibrium.performance_per_phase.time",
                "to_csv.time",
            ],
        )

        self.connect(
            "initialization.initialize_airspeed_time_derivatives.d_vx_dt",
            [
                "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.d_vx_dt",
                "to_csv.d_vx_dt",
            ],
        )

        self.connect(
            "initialization.initialize_airspeed.true_airspeed",
            [
                "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.true_airspeed",
                "solve_equilibrium.compute_dep_equilibrium.compute_dep_effect.true_airspeed",
                "solve_equilibrium.compute_dep_equilibrium.preparation_for_energy_consumption"
                + ".true_airspeed",
                "to_csv.true_airspeed",
            ],
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.compute_dep_effect.delta_Cl",
            "to_csv.delta_Cl",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.compute_dep_effect.delta_Cd",
            "to_csv.delta_Cd",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.compute_dep_effect.delta_Cm",
            "to_csv.delta_Cm",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.alpha", "to_csv.alpha"
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.thrust", "to_csv.thrust"
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.delta_m",
            "to_csv.delta_m",
        )

        self.connect(
            "initialization.initialize_airspeed.equivalent_airspeed", "to_csv.equivalent_airspeed"
        )

        self.connect(
            "solve_equilibrium.update_mass.mass",
            [
                "to_csv.mass",
                "initialization.initialize_airspeed.mass",
            ],
        )

        self.connect(
            "solve_equilibrium.performance_per_phase.fuel_consumed_t",
            [
                "to_csv.fuel_consumed_t",
                "initialization.initialize_center_of_gravity.fuel_consumed_t",
            ],
        )

        self.connect(
            "solve_equilibrium.performance_per_phase.non_consumable_energy_t",
            [
                "to_csv.non_consumable_energy_t",
            ],
        )

        self.connect(
            "solve_equilibrium.performance_per_phase.thrust_rate_t", "to_csv.thrust_rate_t"
        )

        self.connect(
            "initialization.initialize_gamma.gamma",
            ["to_csv.gamma", "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.gamma"],
        )

        self.connect(
            "initialization.initialize_altitude.altitude",
            [
                "to_csv.altitude",
                "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium.altitude",
                "solve_equilibrium.compute_dep_equilibrium.compute_dep_effect.altitude",
                "solve_equilibrium.compute_dep_equilibrium.preparation_for_energy_consumption"
                + ".altitude",
            ],
        )

        self.connect("solve_equilibrium.compute_time_step.time_step", "to_csv.time_step")
