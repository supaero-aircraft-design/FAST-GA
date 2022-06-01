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
import fastoad.api as oad

from ..constants import SUBMODEL_DEP_EFFECT, SUBMODEL_EQUILIBRIUM, SUBMODEL_ENERGY_CONSUMPTION
from ..mission.energy_consumption_preparation import PrepareForEnergyConsumption
from ..mission.equilibrium import Equilibrium


@oad.RegisterSubmodel(SUBMODEL_EQUILIBRIUM, "fastga.submodel.performances.equilibrium.legacy")
class DEPEquilibrium(om.Group):
    """Find the conditions necessary for the aircraft equilibrium."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.LinearBlockGS()

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)
        self.options.declare(
            "promotes_all_variables",
            default=False,
            desc="Set to True if we need to be able to see the flight conditions variables, "
            "not needed in the mission",
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        if self.options["promotes_all_variables"]:
            self.add_subsystem(
                "preparation_for_energy_consumption",
                PrepareForEnergyConsumption(number_of_points=number_of_points),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            self.add_subsystem(
                "compute_equilibrium",
                Equilibrium(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            options_dep = {
                "number_of_points": number_of_points,
                "flaps_position": self.options["flaps_position"],
            }
            self.add_subsystem(
                "compute_dep_effect",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_DEP_EFFECT, options=options_dep),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            options_propulsion = {
                "number_of_points": number_of_points,
                "propulsion_id": self.options["propulsion_id"],
            }
            self.add_subsystem(
                "compute_energy_consumed",
                oad.RegisterSubmodel.get_submodel(
                    SUBMODEL_ENERGY_CONSUMPTION, options=options_propulsion
                ),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
        else:
            self.add_subsystem(
                "preparation_for_energy_consumption",
                PrepareForEnergyConsumption(number_of_points=number_of_points),
                promotes_inputs=["data:*"],
                promotes_outputs=[],
            )
            self.add_subsystem(
                "compute_equilibrium",
                Equilibrium(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes_inputs=["data:*"],
                promotes_outputs=[],
            )
            options_dep = {
                "number_of_points": number_of_points,
                "flaps_position": self.options["flaps_position"],
            }
            self.add_subsystem(
                "compute_dep_effect",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_DEP_EFFECT, options=options_dep),
                promotes_inputs=["data:*"],
                promotes_outputs=[],
            )
            options_propulsion = {
                "number_of_points": number_of_points,
                "propulsion_id": self.options["propulsion_id"],
            }
            self.add_subsystem(
                "compute_energy_consumed",
                oad.RegisterSubmodel.get_submodel(
                    SUBMODEL_ENERGY_CONSUMPTION, options=options_propulsion
                ),
                promotes_inputs=["data:*"],
                promotes_outputs=[],
            )

            self.connect("compute_dep_effect.delta_Cl", "compute_equilibrium.delta_Cl")

            self.connect("compute_dep_effect.delta_Cd", "compute_equilibrium.delta_Cd")

            self.connect("compute_dep_effect.delta_Cm", "compute_equilibrium.delta_Cm")

            self.connect("compute_equilibrium.alpha", "compute_dep_effect.alpha")

            self.connect(
                "compute_equilibrium.thrust",
                "compute_dep_effect.thrust",
            )

            self.connect(
                "preparation_for_energy_consumption.engine_setting_econ",
                "compute_energy_consumed.engine_setting_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.thrust_econ",
                "compute_energy_consumed.thrust_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.altitude_econ",
                "compute_energy_consumed.altitude_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.time_step_econ",
                "compute_energy_consumed.time_step_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.true_airspeed_econ",
                "compute_energy_consumed.true_airspeed_econ",
            )

            self.connect(
                "compute_equilibrium.thrust",
                "preparation_for_energy_consumption.thrust",
            )

        # Solver configuration
        self.nonlinear_solver.options["debug_print"] = False
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-4

        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 50
        self.linear_solver.options["rtol"] = 1e-4
