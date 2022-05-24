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

import numpy as np
import openmdao.api as om

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CRUISE,
    POINTS_NB_CLIMB,
    POINTS_NB_DESCENT,
)


class PerformancePerPhase(om.ExplicitComponent):
    """
    Computes the fuel consumed time spent and ground distance travelled for each phase to
    match the outputs of the previous performance module.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "time", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )
        self.add_input(
            "position", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "fuel_consumed_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="kg",
        )
        self.add_input(
            "non_consumable_energy_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="W*h",
        )
        self.add_input(
            "thrust_rate_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
        )

        self.add_output("data:mission:sizing:main_route:climb:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:climb:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:climb:distance", units="m")
        self.add_output("data:mission:sizing:main_route:climb:duration", units="s")

        self.add_output("data:mission:sizing:main_route:cruise:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:cruise:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:cruise:distance", units="m")
        self.add_output("data:mission:sizing:main_route:cruise:duration", units="s")

        self.add_output("data:mission:sizing:main_route:descent:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:descent:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:descent:distance", units="m")
        self.add_output("data:mission:sizing:main_route:descent:duration", units="s")

        self.add_output("data:mission:sizing:taxi_out:fuel", units="kg")
        self.add_output("data:mission:sizing:taxi_out:energy", units="W*h")
        self.add_output("data:mission:sizing:taxi_in:fuel", units="kg")
        self.add_output("data:mission:sizing:taxi_in:energy", units="W*h")

        self.add_output("fuel_consumed_t", shape=number_of_points, units="kg")
        self.add_output("non_consumable_energy_t", shape=number_of_points, units="W*h")
        self.add_output("thrust_rate_t", shape=number_of_points)

    def setup_partials(self):

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:fuel",
            wrt="fuel_consumed_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:fuel",
            wrt="fuel_consumed_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:fuel",
            wrt="fuel_consumed_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_out:fuel", wrt="fuel_consumed_t_econ", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_in:fuel", wrt="fuel_consumed_t_econ", method="exact"
        )
        self.declare_partials(of="fuel_consumed_t", wrt="fuel_consumed_t_econ", method="exact")

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_out:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_in:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="non_consumable_energy_t", wrt="non_consumable_energy_t_econ", method="exact"
        )

        self.declare_partials(of="thrust_rate_t", wrt="thrust_rate_t_econ", method="exact")

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:distance", wrt="position", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:distance", wrt="position", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:distance", wrt="position", method="exact"
        )

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:duration", wrt="time", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:duration", wrt="time", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:duration", wrt="time", method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        time = inputs["time"]
        position = inputs["position"]
        # This one is two element longer than the other array since it includes the fuel consumed
        # for the taxi phases, hence why we stop at -2 for the descent fuel consumption
        fuel_consumed_t_econ = inputs["fuel_consumed_t_econ"]
        non_consumable_energy = inputs["non_consumable_energy_t_econ"]
        thrust_rate_t_econ = inputs["thrust_rate_t_econ"]

        outputs["data:mission:sizing:main_route:climb:fuel"] = np.sum(
            fuel_consumed_t_econ[0:POINTS_NB_CLIMB]
        )
        outputs["data:mission:sizing:main_route:climb:energy"] = np.sum(
            non_consumable_energy[0:POINTS_NB_CLIMB]
        )
        outputs["data:mission:sizing:main_route:climb:distance"] = max(position[0:POINTS_NB_CLIMB])
        outputs["data:mission:sizing:main_route:climb:duration"] = max(time[0:POINTS_NB_CLIMB])

        outputs["data:mission:sizing:main_route:cruise:fuel"] = np.sum(
            fuel_consumed_t_econ[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE]
        )
        outputs["data:mission:sizing:main_route:cruise:energy"] = np.sum(
            non_consumable_energy[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE]
        )
        outputs["data:mission:sizing:main_route:cruise:distance"] = max(
            position[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE]
        ) - max(position[0:POINTS_NB_CLIMB])
        outputs["data:mission:sizing:main_route:cruise:duration"] = max(
            time[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE]
        ) - max(time[0:POINTS_NB_CLIMB])

        outputs["data:mission:sizing:main_route:descent:fuel"] = np.sum(
            fuel_consumed_t_econ[POINTS_NB_CLIMB + POINTS_NB_CRUISE : -2]
        )
        outputs["data:mission:sizing:main_route:descent:energy"] = np.sum(
            non_consumable_energy[POINTS_NB_CLIMB + POINTS_NB_CRUISE : -2]
        )
        outputs["data:mission:sizing:main_route:descent:distance"] = max(
            position[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]
        ) - max(position[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE])
        outputs["data:mission:sizing:main_route:descent:duration"] = max(
            time[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]
        ) - max(time[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE])

        outputs["data:mission:sizing:taxi_out:fuel"] = fuel_consumed_t_econ[-2]
        outputs["data:mission:sizing:taxi_out:energy"] = non_consumable_energy[-2]
        outputs["data:mission:sizing:taxi_in:fuel"] = fuel_consumed_t_econ[-1]
        outputs["data:mission:sizing:taxi_in:energy"] = non_consumable_energy[-1]

        outputs["fuel_consumed_t"] = fuel_consumed_t_econ[:-2]
        outputs["non_consumable_energy_t"] = non_consumable_energy[:-2]
        outputs["thrust_rate_t"] = thrust_rate_t_econ[:-2]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials[
            "data:mission:sizing:main_route:climb:fuel", "fuel_consumed_t_econ"
        ] = np.concatenate(
            (np.full(POINTS_NB_CLIMB, 1), np.zeros(POINTS_NB_CRUISE + POINTS_NB_DESCENT + 2))
        )
        partials[
            "data:mission:sizing:main_route:climb:energy", "non_consumable_energy_t_econ"
        ] = np.concatenate(
            (np.full(POINTS_NB_CLIMB, 1), np.zeros(POINTS_NB_CRUISE + POINTS_NB_DESCENT + 2))
        )

        partials[
            "data:mission:sizing:main_route:cruise:fuel", "fuel_consumed_t_econ"
        ] = np.concatenate(
            (
                np.zeros(POINTS_NB_CLIMB),
                np.full(POINTS_NB_CRUISE, 1.0),
                np.zeros(POINTS_NB_DESCENT + 2),
            )
        )
        partials[
            "data:mission:sizing:main_route:cruise:energy", "non_consumable_energy_t_econ"
        ] = np.concatenate(
            (
                np.zeros(POINTS_NB_CLIMB),
                np.full(POINTS_NB_CRUISE, 1.0),
                np.zeros(POINTS_NB_DESCENT + 2),
            )
        )

        partials[
            "data:mission:sizing:main_route:cruise:fuel", "fuel_consumed_t_econ"
        ] = np.concatenate(
            (
                np.zeros(POINTS_NB_CLIMB + POINTS_NB_CRUISE),
                np.full(POINTS_NB_DESCENT, 1.0),
                np.zeros(2),
            )
        )
        partials[
            "data:mission:sizing:main_route:cruise:energy", "non_consumable_energy_t_econ"
        ] = np.concatenate(
            (
                np.zeros(POINTS_NB_CLIMB + POINTS_NB_CRUISE),
                np.full(POINTS_NB_DESCENT, 1.0),
                np.zeros(2),
            )
        )

        d_taxi_out_d_fuel = np.zeros(number_of_points + 2)
        d_taxi_out_d_fuel[-2] = 1
        partials["data:mission:sizing:taxi_out:fuel", "fuel_consumed_t_econ"] = d_taxi_out_d_fuel
        partials[
            "data:mission:sizing:taxi_out:energy", "non_consumable_energy_t_econ"
        ] = d_taxi_out_d_fuel

        d_taxi_in_d_fuel = np.zeros(number_of_points + 2)
        d_taxi_in_d_fuel[-1] = 1
        partials["data:mission:sizing:taxi_in:fuel", "fuel_consumed_t_econ"] = d_taxi_in_d_fuel
        partials[
            "data:mission:sizing:taxi_in:energy", "non_consumable_energy_t_econ"
        ] = d_taxi_in_d_fuel

        d_fc_d_fc_t = np.zeros((number_of_points, number_of_points + 2))
        d_fc_d_fc_t[:, :number_of_points] = np.eye(number_of_points)
        partials["fuel_consumed_t", "fuel_consumed_t_econ"] = d_fc_d_fc_t
        partials["non_consumable_energy_t", "non_consumable_energy_t_econ"] = d_fc_d_fc_t

        d_tr_d_tr_t = np.zeros((number_of_points, number_of_points + 2))
        d_tr_d_tr_t[:, :number_of_points] = np.eye(number_of_points)
        partials["thrust_rate_t", "thrust_rate_t_econ"] = d_tr_d_tr_t

        d_climb_d_d_pos = np.zeros(number_of_points)
        d_climb_d_d_pos[POINTS_NB_CLIMB - 1] = 1.0
        partials["data:mission:sizing:main_route:climb:distance", "position"] = d_climb_d_d_pos

        d_cruise_d_d_pos = np.zeros(number_of_points)
        d_cruise_d_d_pos[POINTS_NB_CLIMB + POINTS_NB_CRUISE - 1] = 1.0
        d_cruise_d_d_pos[POINTS_NB_CLIMB - 1] = -1.0
        partials["data:mission:sizing:main_route:cruise:distance", "position"] = d_cruise_d_d_pos

        d_descent_d_d_pos = np.zeros(number_of_points)
        d_descent_d_d_pos[-1] = 1.0
        d_cruise_d_d_pos[POINTS_NB_CLIMB + POINTS_NB_CRUISE - 1] = -1.0
        partials["data:mission:sizing:main_route:descent:distance", "position"] = d_descent_d_d_pos

        partials["data:mission:sizing:main_route:climb:duration", "time"] = d_climb_d_d_pos

        partials["data:mission:sizing:main_route:cruise:duration", "time"] = d_cruise_d_d_pos

        partials["data:mission:sizing:main_route:descent:duration", "time"] = d_descent_d_d_pos
