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
from stdatm import Atmosphere

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
)


class InitializeAirspeedDerivatives(om.ExplicitComponent):
    """Computes the d_vx_dt at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "true_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "equivalent_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "gamma", shape=number_of_points, val=np.full(number_of_points, np.nan), units="deg"
        )

        self.add_output(
            "d_vx_dt", shape=number_of_points, val=np.full(number_of_points, 0.0), units="m/s**2"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        true_airspeed = inputs["true_airspeed"]
        equivalent_airspeed = inputs["equivalent_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"] * np.pi / 180.0

        altitude_climb = altitude[0:POINTS_NB_CLIMB]
        gamma_climb = gamma[0:POINTS_NB_CLIMB]
        equivalent_airspeed_climb = equivalent_airspeed[0:POINTS_NB_CLIMB]
        true_airspeed_climb = true_airspeed[0:POINTS_NB_CLIMB]

        altitude_descent = altitude[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]
        gamma_descent = gamma[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]
        equivalent_airspeed_descent = equivalent_airspeed[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]
        true_airspeed_descent = true_airspeed[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]

        atm_climb_plus_1 = Atmosphere(altitude_climb + 1.0, altitude_in_feet=False)
        atm_climb_plus_1.equivalent_airspeed = equivalent_airspeed_climb
        d_v_tas_dh_climb = atm_climb_plus_1.true_airspeed - true_airspeed_climb
        d_vx_dt_climb = d_v_tas_dh_climb * true_airspeed_climb * np.sin(gamma_climb)

        atm_descent_plus_1 = Atmosphere(altitude_descent + 1.0, altitude_in_feet=False)
        atm_descent_plus_1.equivalent_airspeed = equivalent_airspeed_descent
        d_v_tas_dh_descent = atm_descent_plus_1.true_airspeed - true_airspeed_descent
        d_vx_dt_descent = d_v_tas_dh_descent * true_airspeed_descent * np.sin(gamma_descent)

        d_vx_dt = np.concatenate((d_vx_dt_climb, np.zeros(POINTS_NB_CRUISE), d_vx_dt_descent))

        outputs["d_vx_dt"] = d_vx_dt
