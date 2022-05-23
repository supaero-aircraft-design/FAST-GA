"""FAST - Copyright (c) 2021 ONERA ISAE."""

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

import openmdao.api as om
import numpy as np

from stdatm import Atmosphere
from scipy.constants import g

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
)


class InitializeAirspeed(om.ExplicitComponent):
    """Computes the fuel consumed at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)

        self.add_input("mass", val=np.full(n, np.nan), shape=n, units="kg")
        self.add_input("altitude", val=np.full(n, np.nan), shape=n, units="m")

        self.add_output("true_airspeed", val=np.full(n, 50.0), units="m/s")
        self.add_output("equivalent_airspeed", val=np.full(n, 50.0), units="m/s")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        v_tas_cruise = inputs["data:TLAR:v_cruise"]

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]

        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mass = inputs["mass"]
        altitude = inputs["altitude"]

        altitude_climb = altitude[0:POINTS_NB_CLIMB]
        altitude_cruise = altitude[POINTS_NB_CLIMB : POINTS_NB_CLIMB + POINTS_NB_CRUISE]
        altitude_descent = altitude[POINTS_NB_CLIMB + POINTS_NB_CRUISE :]

        # Computes the airspeed that gives the best climb rate
        # FIXME: VCAS constant-speed strategy is specific to ICE-propeller configuration,
        # FIXME: could be an input!
        cl = np.sqrt(3 * cd0 / coeff_k_wing)
        atm_climb = Atmosphere(altitude_climb, altitude_in_feet=False)

        vs1 = np.sqrt((mass[0] * g) / (0.5 * atm_climb.density[0] * wing_area * cl_max_clean))
        # Using the denomination in Gudmundsson
        vy = np.sqrt((mass[0] * g) / (0.5 * atm_climb.density[0] * wing_area * cl))
        v_eas_climb = max(vy, 1.3 * vs1)
        atm_climb.equivalent_airspeed = np.full_like(altitude_climb, v_eas_climb)

        true_airspeed_climb = atm_climb.true_airspeed

        atm_cruise = Atmosphere(altitude_cruise[0], altitude_in_feet=False)
        atm_cruise.true_airspeed = v_tas_cruise
        true_airspeed_cruise = np.full_like(altitude_cruise, v_tas_cruise)
        equivalent_airspeed_cruise = np.full_like(altitude_cruise, atm_cruise.equivalent_airspeed)

        cl_opt = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]

        mass_descent = mass[POINTS_NB_CLIMB + POINTS_NB_CRUISE + 1]
        atm_descent = Atmosphere(altitude_descent, altitude_in_feet=False)
        vs1 = np.sqrt(
            (mass_descent * g) / (0.5 * atm_descent.density[0] * wing_area * cl_max_clean)
        )
        v_eas_descent = max(
            np.sqrt((mass_descent * g) / (0.5 * atm_descent.density[0] * wing_area * cl_opt)),
            1.3 * vs1,
        )

        atm_descent.equivalent_airspeed = np.full_like(altitude_descent, v_eas_descent)
        true_airspeed_descent = atm_descent.true_airspeed

        outputs["true_airspeed"] = np.concatenate(
            (true_airspeed_climb, true_airspeed_cruise, true_airspeed_descent)
        )
        outputs["equivalent_airspeed"] = np.concatenate(
            (
                atm_climb.equivalent_airspeed,
                equivalent_airspeed_cruise,
                atm_descent.equivalent_airspeed,
            )
        )
