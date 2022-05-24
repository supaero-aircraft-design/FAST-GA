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
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
)


class InitializeTimeAndDistance(om.ExplicitComponent):
    """Initializes time and ground distance at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        # Cannot use the vertical speed vector previously computed since it is gonna be
        # initialized at 0.0 which will cause a problem for the time computation
        self.add_input("data:TLAR:range", np.nan, units="m")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:sea_level", val=np.nan, units="m/s"
        )
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:cruise_level", val=np.nan, units="m/s"
        )
        self.add_input("data:mission:sizing:main_route:descent:descent_rate", np.nan, units="m/s")

        self.add_input(
            "true_airspeed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )
        self.add_input(
            "horizontal_speed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )
        self.add_input(
            "altitude", val=np.full(number_of_points, np.nan), shape=number_of_points, units="m"
        )

        self.add_output("time", val=np.linspace(0.0, 7200.0, number_of_points), units="s")
        self.add_output("position", val=np.linspace(0.0, 926000.0, number_of_points), units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        altitude = inputs["altitude"]
        horizontal_speed = inputs["horizontal_speed"]

        mission_range = inputs["data:TLAR:range"]
        v_tas_cruise = inputs["data:TLAR:v_cruise"]

        climb_rate_sl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"])
        climb_rate_cl = float(
            inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"]
        )
        descent_rate = -abs(inputs["data:mission:sizing:main_route:descent:descent_rate"])

        altitude_climb = altitude[0:POINTS_NB_CLIMB]
        horizontal_speed_climb = horizontal_speed[0:POINTS_NB_CLIMB]
        altitude_descent = altitude[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]
        horizontal_speed_descent = horizontal_speed[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]

        # Computing the time evolution during the climb phase, based on the altitude sampling and
        # the desired climb rate
        mid_altitude_climb = (altitude_climb[:-1] + altitude_climb[1:]) / 2.0
        mid_climb_rate = np.interp(
            mid_altitude_climb, [0.0, max(altitude_climb)], [climb_rate_sl, climb_rate_cl]
        )
        mid_horizontal_speed_climb = (
            horizontal_speed_climb[:-1] + horizontal_speed_climb[1:]
        ) / 2.0
        altitude_step_climb = altitude_climb[1:] - altitude_climb[:-1]
        time_to_climb_step = altitude_step_climb / mid_climb_rate
        position_increment_climb = mid_horizontal_speed_climb * time_to_climb_step

        time_climb = np.concatenate((np.array([0]), np.cumsum(time_to_climb_step)))
        position_climb = np.concatenate((np.array([0]), np.cumsum(position_increment_climb)))

        # Computing the time evolution during the descent phase, based on the altitude sampling and
        # the desired descent rate
        mid_descent_rate = np.full_like(altitude_descent[1:], abs(descent_rate))
        mid_horizontal_speed_descent = (
            horizontal_speed_descent[:-1] + horizontal_speed_descent[1:]
        ) / 2.0
        altitude_step_descent = abs(altitude_descent[1:] - altitude_descent[:-1])
        time_to_descend_step = altitude_step_descent / mid_descent_rate
        position_increment_descent = mid_horizontal_speed_descent * time_to_descend_step

        time_descent = np.concatenate((np.array([0]), np.cumsum(time_to_descend_step)))
        position_descent = np.concatenate((np.array([0]), np.cumsum(position_increment_descent)))

        # Cruise position computation
        cruise_range = mission_range - position_climb[-1] - position_descent[-1]
        cruise_distance_step = cruise_range / (POINTS_NB_CRUISE + 1)
        position_cruise = np.linspace(
            position_climb[-1] + cruise_distance_step,
            position_climb[-1] + cruise_range - cruise_distance_step,
            POINTS_NB_CRUISE,
        )[:, 0]

        cruise_time_array = (position_cruise - position_climb[-1]) / v_tas_cruise + time_climb[-1]
        cruise_time = cruise_range / v_tas_cruise

        position_descent += position_climb[-1] + cruise_range
        time_descent += time_climb[-1] + cruise_time

        position = np.concatenate((position_climb, position_cruise, position_descent))
        time = np.concatenate((time_climb, cruise_time_array, time_descent))

        outputs["position"] = position
        outputs["time"] = time
