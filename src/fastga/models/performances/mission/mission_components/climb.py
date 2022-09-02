"""Simple module for climb computation."""
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

import os
import logging
import time
import numpy as np
import openmdao.api as om

from scipy.constants import g
from scipy.interpolate import interp1d

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting

from stdatm import Atmosphere

from fastga.models.performances.mission.takeoff import SAFETY_HEIGHT

from ..dynamic_equilibrium import DynamicEquilibrium, save_df
from ..constants import SUBMODEL_CLIMB, SUBMODEL_CLIMB_SPEED

_LOGGER = logging.getLogger(__name__)

POINTS_NB_CLIMB = 100
MAX_CALCULATION_TIME = 15  # time in seconds

oad.RegisterSubmodel.active_models[
    SUBMODEL_CLIMB
] = "fastga.submodel.performances.mission.climb.legacy"
oad.RegisterSubmodel.active_models[
    SUBMODEL_CLIMB_SPEED
] = "fastga.submodel.performances.mission.climb_speed.legacy"


@oad.RegisterSubmodel(SUBMODEL_CLIMB, "fastga.submodel.performances.mission.climb.legacy")
class ComputeClimb(DynamicEquilibrium):
    """
    Compute the fuel consumption on climb segment with constant VCAS and a climb rate which
    varies linearly with the altitude.
    The hypothesis of small alpha/gamma angles is done.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        super().initialize()
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        super().setup()
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:sea_level", val=np.nan, units="m/s"
        )
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:cruise_level", val=np.nan, units="m/s"
        )
        self.add_input("data:mission:sizing:main_route:climb:v_cas", val=np.nan, units="m/s")

        self.add_output("data:mission:sizing:main_route:climb:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:climb:distance", units="m")
        self.add_output("data:mission:sizing:main_route:climb:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Delete previous .csv results
        if self.options["out_file"] != "":
            # noinspection PyBroadException
            flight_point_df = None
            try:
                os.remove(self.options["out_file"])
            except:
                _LOGGER.info("Failed to remove %s file!", self.options["out_file"])

        propulsion_model = self._engine_wrapper.get_model(inputs)
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        v_cas = inputs["data:mission:sizing:main_route:climb:v_cas"]
        climb_rate_sl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"])
        climb_rate_cl = float(
            inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"]
        )

        # Define initial conditions
        t_start = time.time()
        altitude_t = SAFETY_HEIGHT  # conversion to m
        distance_t = 0.0
        time_t = 0.0
        mass_t = mtow - (m_to + m_tk + m_ic)
        mass_fuel_t = 0.0
        previous_step = ()

        # Calculate constant speed (cos(gamma)~1) and corresponding climb angle
        atm = Atmosphere(altitude_t, altitude_in_feet=False)
        atm.calibrated_airspeed = v_cas
        v_tas = atm.true_airspeed
        gamma = np.arcsin(climb_rate_sl / v_tas)

        # Define specific time step ~POINTS_NB_CLIMB points for calculation (with ground conditions)
        time_step = ((cruise_altitude - SAFETY_HEIGHT) / climb_rate_sl) / float(POINTS_NB_CLIMB)

        while altitude_t < cruise_altitude:

            # Calculate dynamic pressure
            atm = Atmosphere(altitude_t, altitude_in_feet=False)
            atm.calibrated_airspeed = v_cas
            v_tas = atm.true_airspeed
            climb_rate = interp1d([0.0, float(cruise_altitude)], [climb_rate_sl, climb_rate_cl])(
                altitude_t
            )
            gamma = np.arcsin(climb_rate / v_tas)
            mach = v_tas / atm.speed_of_sound
            atm_1 = Atmosphere(altitude_t + 1.0, altitude_in_feet=False)
            atm_1.calibrated_airspeed = v_cas
            dv_tas_dh = atm_1.true_airspeed - v_tas
            dvx_dt = dv_tas_dh * v_tas * np.sin(gamma)
            dynamic_pressure = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium
            previous_step = self.dynamic_equilibrium(
                inputs, gamma, dynamic_pressure, dvx_dt, 0.0, mass_t, "none", previous_step[0:2]
            )
            thrust = float(previous_step[1])

            # Compute consumption
            flight_point = oad.FlightPoint(
                mach=mach,
                altitude=altitude_t,
                engine_setting=EngineSetting.CLIMB,
                thrust_is_regulated=True,
                thrust=thrust,
            )
            propulsion_model.compute_flight_points(flight_point)
            if flight_point.thrust_rate > 1.0:
                _LOGGER.warning("Thrust rate is above 1.0, value clipped at 1.0")

            # Save results
            if self.options["out_file"] != "":
                flight_point_df = save_df(
                    time_t,
                    altitude_t,
                    distance_t,
                    mass_t,
                    v_tas,
                    v_cas,
                    atm.density,
                    gamma * 180.0 / np.pi,
                    previous_step,
                    flight_point.thrust_rate,
                    flight_point.sfc,
                    "sizing:main_route:climb",
                    flight_point_df,
                )

            consumed_mass_1s = propulsion_model.get_consumed_mass(flight_point, 1.0)

            # Calculate distance variation (earth axis)
            v_z = v_tas * np.sin(gamma)
            v_x = v_tas * np.cos(gamma)
            time_step = min(time_step, (cruise_altitude - altitude_t) / v_z)
            altitude_t += v_z * time_step
            distance_t += v_x * time_step

            # Estimate mass evolution and update time
            mass_fuel_t += consumed_mass_1s * time_step
            mass_t = mass_t - consumed_mass_1s * time_step
            time_t += time_step

            # Check calculation duration
            if (time.time() - t_start) > MAX_CALCULATION_TIME:
                raise Exception(
                    "Time calculation duration for climb phase [%f s] exceeded!"
                    % MAX_CALCULATION_TIME
                )

        # Save results
        if self.options["out_file"] != "":
            flight_point_df = save_df(
                time_t,
                altitude_t,
                distance_t,
                mass_t,
                v_tas,
                v_cas,
                atm.density,
                gamma * 180.0 / np.pi,
                previous_step,
                flight_point.thrust_rate,
                flight_point.sfc,
                "sizing:main_route:climb",
                flight_point_df,
            )
            self.save_csv(flight_point_df)

        outputs["data:mission:sizing:main_route:climb:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:climb:distance"] = distance_t
        outputs["data:mission:sizing:main_route:climb:duration"] = time_t


@oad.RegisterSubmodel(
    SUBMODEL_CLIMB_SPEED, "fastga.submodel.performances.mission.climb_speed.legacy"
)
class ComputeClimbSpeed(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)

        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")

        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")

        self.add_output("data:mission:sizing:main_route:climb:v_cas", units="m/s")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        altitude_t = SAFETY_HEIGHT  # conversion to m

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        c_l_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]

        mass_t = mtow - (m_to + m_tk + m_ic)

        c_l = np.sqrt(3 * cd0 / coeff_k_wing)
        atm = Atmosphere(altitude_t, altitude_in_feet=False)
        vs1 = np.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * c_l_max_clean))
        v_cas = max(np.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * c_l)), 1.3 * vs1)

        outputs["data:mission:sizing:main_route:climb:v_cas"] = v_cas
