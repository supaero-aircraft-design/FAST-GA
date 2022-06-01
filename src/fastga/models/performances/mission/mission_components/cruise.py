"""Simple module for cruise computation."""
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
import time
import numpy as np

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting

from stdatm import Atmosphere

from ..dynamic_equilibrium import DynamicEquilibrium, save_df
from ..constants import SUBMODEL_CRUISE

_LOGGER = logging.getLogger(__name__)

POINTS_NB_CRUISE = 100
MAX_CALCULATION_TIME = 15  # time in seconds

oad.RegisterSubmodel.active_models[
    SUBMODEL_CRUISE
] = "fastga.submodel.performances.mission.cruise.legacy"


@oad.RegisterSubmodel(SUBMODEL_CRUISE, "fastga.submodel.performances.mission.cruise.legacy")
class ComputeCruise(DynamicEquilibrium):
    """
    Compute the fuel consumption on cruise segment with constant VTAS and altitude.
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

        self.add_input("data:TLAR:range", np.nan, units="m")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:descent:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:climb:duration", np.nan, units="s")

        self.add_output("data:mission:sizing:main_route:cruise:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:cruise:distance", units="m")
        self.add_output("data:mission:sizing:main_route:cruise:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["out_file"] != "":
            # noinspection PyBroadException
            flight_point_df = None

        propulsion_model = self._engine_wrapper.get_model(inputs)
        v_tas = inputs["data:TLAR:v_cruise"]
        cruise_distance = max(
            0.0,
            (
                inputs["data:TLAR:range"]
                - inputs["data:mission:sizing:main_route:climb:distance"]
                - inputs["data:mission:sizing:main_route:descent:distance"]
            ),
        )
        if cruise_distance == 0.0:
            _LOGGER.warning(
                "Cruise distance is negative, check the input value mainly the range "
                "and/or the climb and descent inputs"
            )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]

        # Define specific time step ~POINTS_NB_CRUISE points for calculation
        time_step = (cruise_distance / v_tas) / float(POINTS_NB_CRUISE)

        # Define initial conditions
        t_start = time.time()
        distance_t = 0.0
        time_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_tk + m_ic + m_cl)
        atm = Atmosphere(cruise_altitude, altitude_in_feet=False)
        atm.true_airspeed = v_tas
        mach = atm.mach
        previous_step = ()

        while distance_t < cruise_distance:

            # Calculate dynamic pressure
            dynamic_pressure = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium
            previous_step = self.dynamic_equilibrium(
                inputs, 0.0, dynamic_pressure, 0.0, 0.0, mass_t, "none", previous_step[0:2]
            )
            thrust = float(previous_step[1])

            # Compute consumption
            flight_point = oad.FlightPoint(
                mach=mach,
                altitude=cruise_altitude,
                engine_setting=EngineSetting.CRUISE,
                thrust_is_regulated=True,
                thrust=thrust,
            )
            propulsion_model.compute_flight_points(flight_point)
            if flight_point.thrust_rate > 1.0:
                _LOGGER.warning("Thrust rate is above 1.0, value clipped at 1.0")

            # Save results
            if self.options["out_file"] != "":
                flight_point_df = save_df(
                    time_t + inputs["data:mission:sizing:main_route:climb:duration"],
                    cruise_altitude,
                    distance_t + inputs["data:mission:sizing:main_route:climb:distance"],
                    mass_t,
                    v_tas,
                    atm.calibrated_airspeed,
                    atm.density,
                    0.0,
                    previous_step,
                    flight_point.thrust_rate,
                    flight_point.sfc,
                    "sizing:main_route:cruise",
                    flight_point_df,
                )

            consumed_mass_1s = propulsion_model.get_consumed_mass(flight_point, 1.0)
            # Calculate distance increase
            distance_t += v_tas * min(time_step, (cruise_distance - distance_t) / v_tas)

            # Estimate mass evolution and update time
            mass_fuel_t += consumed_mass_1s * min(time_step, (cruise_distance - distance_t) / v_tas)
            mass_t = mass_t - consumed_mass_1s * min(
                time_step, (cruise_distance - distance_t) / v_tas
            )
            time_t += min(time_step, (cruise_distance - distance_t) / v_tas)

            # Check calculation duration
            if (time.time() - t_start) > MAX_CALCULATION_TIME:
                raise Exception(
                    "Time calculation duration for cruise phase [%f s] exceeded!"
                    % MAX_CALCULATION_TIME
                )

        # Save results
        if self.options["out_file"] != "":
            flight_point_df = save_df(
                time_t + inputs["data:mission:sizing:main_route:climb:duration"],
                cruise_altitude,
                distance_t + inputs["data:mission:sizing:main_route:climb:distance"],
                mass_t,
                v_tas,
                atm.calibrated_airspeed,
                atm.density,
                0.0,
                previous_step,
                flight_point.thrust_rate,
                flight_point.sfc,
                "sizing:main_route:cruise",
                flight_point_df,
            )
            self.save_csv(flight_point_df)

        outputs["data:mission:sizing:main_route:cruise:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:cruise:distance"] = distance_t
        outputs["data:mission:sizing:main_route:cruise:duration"] = time_t
