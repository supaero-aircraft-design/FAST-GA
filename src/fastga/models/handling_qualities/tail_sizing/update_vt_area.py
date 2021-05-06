"""
Estimation of vertical tail area
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

import numpy as np
import openmdao.api as om

from fastoad.model_base import Atmosphere, FlightPoint
from fastoad.model_base.propulsion import FuelEngineSet
# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting


class UpdateVTArea(om.ExplicitComponent):
    """
    Computes needed vt area to:
      - have enough rotational moment/controllability during cruise
      - compensate 1-failed engine linear trajectory at limited altitude (5000ft)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input("data:aerodynamics:fuselage:cruise:CnBeta", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:area", val=2.5, units="m**2")

        self.declare_partials(
            "*",
            "*",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Sizing constraints for the vertical tail.
        # Limiting cases: rotating torque objective (cn_beta_goal) during cruise, and  
        # compensation of engine failure induced torque at approach speed/altitude. 
        # Returns maximum area.

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        engine_number = inputs["data:geometry:propulsion:count"]
        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_mac_position = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        cn_beta_fuselage = inputs["data:aerodynamics:fuselage:cruise:CnBeta"]
        cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"]
        cruise_speed = inputs["data:TLAR:v_cruise"]
        approach_speed = inputs["data:TLAR:v_approach"]
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        wing_htp_distance = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        nac_wet_area = inputs["data:geometry:propulsion:nacelle:wet_area"]
        y_nacelle = inputs["data:geometry:propulsion:nacelle:y"]

        # CASE1: OBJECTIVE TORQUE @ CRUISE #############################################################################

        atm = Atmosphere(cruise_altitude)
        speed_of_sound = atm.speed_of_sound
        cruise_mach = cruise_speed / speed_of_sound
        # Matches suggested goal by Raymer, Fig 16.20
        cn_beta_goal = 0.0569 - 0.01694 * cruise_mach + 0.15904 * cruise_mach ** 2

        required_cnbeta_vtp = cn_beta_goal - cn_beta_fuselage
        distance_to_cg = wing_htp_distance + 0.25 * l0_wing - cg_mac_position * l0_wing
        area_1 = required_cnbeta_vtp / (distance_to_cg / wing_area / span * cl_alpha_vt)

        # CASE2: ENGINE FAILURE COMPENSATION DURING CLIMB ##############################################################

        failure_altitude = 5000.0  # CS23 for Twin engine - at 5000ft
        atm = Atmosphere(failure_altitude)
        speed_of_sound = atm.speed_of_sound
        pressure = atm.pressure
        if engine_number == 2.0:
            stall_speed = approach_speed / 1.3
            mc_speed = 1.2 * stall_speed  # Flights mechanics from GA - Serge Bonnet CS23
            mc_mach = mc_speed / speed_of_sound
            # Calculation of engine power for given conditions
            flight_point = FlightPoint(
                mach=mc_mach, altitude=failure_altitude, engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0
            )  # forced to maximum thrust
            propulsion_model.compute_flight_points(flight_point)
            thrust = float(flight_point.thrust)
            # Calculation of engine thrust and nacelle drag (failed one)
            nac_drag = 0.07 * nac_wet_area  # FIXME: the form factor should not be fixed outside propulsion module!
            # Torque compensation
            area_2 = (
                    2 * (y_nacelle / wing_htp_distance) * (thrust + nac_drag)
                    / (pressure * mc_mach ** 2 * 0.9 * 0.42 * 10)
            )
        else:
            area_2 = 0.0

        outputs["data:geometry:vertical_tail:area"] = max(area_1, area_2)
