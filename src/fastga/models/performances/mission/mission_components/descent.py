"""Simple module for descent computation."""
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

import copy
import logging
import time
import numpy as np
import openmdao.api as om

from scipy.constants import g

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting

from stdatm import Atmosphere


from ..dynamic_equilibrium import DynamicEquilibrium, save_df
from ..constants import SUBMODEL_DESCENT, SUBMODEL_DESCENT_SPEED

_LOGGER = logging.getLogger(__name__)

POINTS_NB_DESCENT = 50
MAX_CALCULATION_TIME = 15  # time in seconds

oad.RegisterSubmodel.active_models[
    SUBMODEL_DESCENT
] = "fastga.submodel.performances.mission.descent.legacy"
oad.RegisterSubmodel.active_models[
    SUBMODEL_DESCENT_SPEED
] = "fastga.submodel.performances.mission.descent_speed.legacy"


@oad.RegisterSubmodel(SUBMODEL_DESCENT, "fastga.submodel.performances.mission.descent.legacy")
class ComputeDescent(DynamicEquilibrium):
    """
    Compute the fuel consumption on descent segment with constant VCAS and descent
    rate.
    The hypothesis of small alpha angle is done.
    Warning: Descent rate is reduced if cd/cl < abs(desc_rate)!
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

        self.add_input("data:mission:sizing:main_route:descent:descent_rate", np.nan, units="m/s")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:climb:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:cruise:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:descent:v_cas", np.nan, units="m/s")

        self.add_output("data:mission:sizing:main_route:descent:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:descent:distance", 0.0, units="m")
        self.add_output("data:mission:sizing:main_route:descent:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["out_file"] != "":
            # noinspection PyBroadException
            flight_point_df = None

        propulsion_model = self._engine_wrapper.get_model(inputs)
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        descent_rate = inputs["data:mission:sizing:main_route:descent:descent_rate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cr = inputs["data:mission:sizing:main_route:cruise:fuel"]
        v_cas = inputs["data:mission:sizing:main_route:descent:v_cas"]

        # Define initial conditions
        t_start = time.time()
        altitude_t = copy.deepcopy(cruise_altitude)
        distance_t = 0.0
        time_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_tk + m_ic + m_cl + m_cr)
        previous_step = ()

        # Calculate constant speed (cos(gamma)~1) and corresponding descent angle
        # FIXME: VCAS constant-speed strategy is specific to ICE-propeller configuration, should be
        # FIXME: an input!
        atm = Atmosphere(altitude_t, altitude_in_feet=False)
        atm.calibrated_airspeed = v_cas
        v_tas = atm.true_airspeed
        gamma = np.arcsin(descent_rate / v_tas)

        # Define specific time step ~POINTS_NB_DESCENT points for calculation (with ground
        # conditions)
        time_step = abs((altitude_t / descent_rate)) / float(POINTS_NB_DESCENT)

        while altitude_t > 0.0:

            # Calculate dynamic pressure
            atm = Atmosphere(altitude_t, altitude_in_feet=False)
            atm.calibrated_airspeed = v_cas
            v_tas = atm.true_airspeed
            mach = v_tas / atm.speed_of_sound
            gamma = np.arcsin(descent_rate / v_tas)
            atm_1 = Atmosphere(altitude_t + 1.0, altitude_in_feet=False)
            atm_1.calibrated_airspeed = v_cas
            dv_tas_dh = atm_1.true_airspeed - v_tas
            dvx_dt = dv_tas_dh * v_tas * np.sin(gamma)
            dynamic_pressure = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium, decrease gamma if obtained thrust is negative
            previous_step = self.dynamic_equilibrium(
                inputs, gamma, dynamic_pressure, dvx_dt, 0.0, mass_t, "none", previous_step[0:2]
            )
            thrust = previous_step[1]
            while thrust < 0.0:
                gamma = 0.9 * gamma
                previous_step = self.dynamic_equilibrium(
                    inputs, gamma, dynamic_pressure, dvx_dt, 0.0, mass_t, "none", previous_step[0:2]
                )
                thrust = previous_step[1]

            # Compute consumption
            # FIXME: DESCENT setting on engine does not exist, replaced by CLIMB for test
            flight_point = oad.FlightPoint(
                mach=mach,
                altitude=altitude_t,
                engine_setting=EngineSetting.CLIMB,
                thrust_is_regulated=True,
                thrust=thrust,
            )
            propulsion_model.compute_flight_points(flight_point)
            # Save results
            if self.options["out_file"] != "":
                flight_point_df = save_df(
                    time_t
                    + inputs["data:mission:sizing:main_route:climb:duration"]
                    + inputs["data:mission:sizing:main_route:cruise:duration"],
                    altitude_t,
                    distance_t
                    + inputs["data:mission:sizing:main_route:climb:distance"]
                    + inputs["data:mission:sizing:main_route:cruise:distance"],
                    mass_t,
                    v_tas,
                    v_cas,
                    atm.density,
                    gamma * 180.0 / np.pi,
                    previous_step,
                    flight_point.thrust_rate,
                    flight_point.sfc,
                    "sizing:main_route:descent",
                    flight_point_df,
                )
            consumed_mass_1s = propulsion_model.get_consumed_mass(flight_point, 1.0)

            # Calculate distance variation (earth axis)
            v_x = v_tas * np.cos(gamma)
            v_z = v_tas * np.sin(gamma)
            time_step = min(time_step, -altitude_t / v_z)
            distance_t += v_x * time_step
            altitude_t += v_z * time_step

            # Estimate mass evolution and update time
            mass_fuel_t += consumed_mass_1s * time_step
            mass_t = mass_t - consumed_mass_1s * time_step
            time_t += time_step

            # Check calculation duration
            if (time.time() - t_start) > MAX_CALCULATION_TIME:
                raise Exception(
                    "Time calculation duration for descent phase [%f s] exceeded!"
                    % MAX_CALCULATION_TIME
                )

        # Save results
        if self.options["out_file"] != "":
            flight_point_df = save_df(
                time_t
                + inputs["data:mission:sizing:main_route:climb:duration"]
                + inputs["data:mission:sizing:main_route:cruise:duration"],
                altitude_t,
                distance_t
                + inputs["data:mission:sizing:main_route:climb:distance"]
                + inputs["data:mission:sizing:main_route:cruise:distance"],
                mass_t,
                v_tas,
                v_cas,
                atm.density,
                gamma * 180.0 / np.pi,
                previous_step,
                flight_point.thrust_rate,
                flight_point.sfc,
                "sizing:main_route:descent",
                flight_point_df,
            )
            self.save_csv(flight_point_df)

        outputs["data:mission:sizing:main_route:descent:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:descent:distance"] = distance_t
        outputs["data:mission:sizing:main_route:descent:duration"] = time_t


@oad.RegisterSubmodel(
    SUBMODEL_DESCENT_SPEED, "fastga.submodel.performances.mission.descent_speed.legacy"
)
class ComputeDescentSpeed(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)

        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")

        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output("data:mission:sizing:main_route:descent:v_cas", units="m/s")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cr = inputs["data:mission:sizing:main_route:cruise:fuel"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]

        c_l = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]
        c_l_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]

        mass_t = mtow - (m_to + m_tk + m_ic + m_cl + m_cr)

        altitude_t = copy.deepcopy(cruise_altitude)

        atm = Atmosphere(altitude_t, altitude_in_feet=False)
        vs1 = np.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * c_l_max_clean))
        v_cas = max(np.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * c_l)), 1.3 * vs1)

        outputs["data:mission:sizing:main_route:descent:v_cas"] = v_cas
