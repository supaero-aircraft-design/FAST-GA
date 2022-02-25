"""Simple module for complete mission."""
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
import os
import math
import openmdao.api as om
import copy
import logging
from typing import Sequence, Union

from scipy.constants import g
from scipy.interpolate import interp1d
import time

from fastoad.model_base import Atmosphere, FlightPoint

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain

from fastga.models.performances.takeoff_HE import SAFETY_HEIGHT, TakeOffPhase
from fastga.models.performances.dynamic_equilibrium import DynamicEquilibrium

from fastga.models.propulsion.hybrid_propulsion.base import HybridEngineSet
from fastga.models.weight.cg.cg_variation import InFlightCGVariation

POINTS_NB_CLIMB = 100
POINTS_NB_CRUISE = 100
POINTS_NB_DESCENT = 50
MAX_CALCULATION_TIME = 15  # time in seconds
POINTS_POWER_COUNT = 200

_LOGGER = logging.getLogger(__name__)


@RegisterOpenMDAOSystem("fastga.performances.mission_HE", domain=ModelDomain.PERFORMANCE)
class Mission_HE(om.Group):
    """
    Computes analytically the hydrogen mass and the battery energy necessary for each part of the flight cycle.

    Loop on the distance crossed during descent and cruise distance/fuel mass and battery energy.

    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)
        self.options.declare("out_file", default="", types=str)

    def setup(self):
        self.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
        self.add_subsystem(
            "taxi_out",
            _compute_taxi(propulsion_id=self.options["propulsion_id"], taxi_out=True, ),
            promotes=["*"],
        )
        # self.add_subsystem(
        #     "takeoff", TakeOffPhase(propulsion_id=self.options["propulsion_id"]), promotes=["*"]
        # )
        self.add_subsystem(
            "climb",
            _compute_climb(
                propulsion_id=self.options["propulsion_id"], out_file=self.options["out_file"],
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "cruise",
            _compute_cruise(
                propulsion_id=self.options["propulsion_id"], out_file=self.options["out_file"],
            ),
            promotes=["*"],
        )
        self.add_subsystem("reserve", _compute_reserve(), promotes=["*"])
        self.add_subsystem(
            "descent",
            _compute_descent(
                propulsion_id=self.options["propulsion_id"], out_file=self.options["out_file"],
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "taxi_in",
            _compute_taxi(propulsion_id=self.options["propulsion_id"], taxi_out=False, ),
            promotes=["*"],
        )
        self.add_subsystem("update_resources", UpdateResources(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["debug_print"] = True
        # self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        # self.nonlinear_solver.options["reraise_child_analysiserror"] = True
        self.nonlinear_solver.options["rtol"] = 1e-2

        self.linear_solver = om.LinearBlockGS()
        # self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10
        self.linear_solver.options["rtol"] = 1e-2


class _compute_reserve(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:reserve:duration", np.nan, units="s")

        self.add_input("data:mission:sizing:main_route:reserve:battery_power", np.nan, units="kW")
        self.add_input("settings:electrical_system:system_voltage", np.nan, units="V")

        self.add_output("data:mission:sizing:main_route:reserve:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:reserve:battery_capacity", units="A*h")
        self.add_output("data:mission:sizing:main_route:reserve:battery_energy", units="kW*h")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m_reserve = (
                inputs["data:mission:sizing:main_route:cruise:fuel"]
                * inputs["data:mission:sizing:main_route:reserve:duration"]
                / max(
            1e-6, inputs["data:mission:sizing:main_route:cruise:duration"]
        )  # Avoid 0 division
        )

        energy_reserve = (
                inputs["data:mission:sizing:main_route:reserve:battery_power"]
                * inputs["data:mission:sizing:main_route:reserve:duration"] / 3600
        )

        capacity_reserve = (
                energy_reserve * 1000
                / max(1e-6, inputs["settings:electrical_system:system_voltage"])  # Avoid 0 division
        )

        outputs["data:mission:sizing:main_route:reserve:fuel"] = m_reserve
        outputs["data:mission:sizing:main_route:reserve:battery_capacity"] = capacity_reserve
        outputs["data:mission:sizing:main_route:reserve:battery_energy"] = energy_reserve


class UpdateResources(om.ExplicitComponent):
    """
    Computes the total mass of hydrogen, total battery energy and capacity required to complete the mission.
    Also used to compute the max current at any point during the mission. The max current
    can be used to size the power electronics (alternatively, max power delivered by the engine divided the voltage
    of the electrical system gives the max current that can be used to size the power electronics).
    Based on : FAST-GA-ELEC
    """

    def setup(self):

        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:reserve:fuel", np.nan, units="kg")
        # self.add_input("data:mission:sizing:main_route:descent:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_in:fuel", np.nan, units="kg")

        self.add_input("data:mission:sizing:taxi_out:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:takeoff:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:initial_climb:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:main_route:climb:battery_current", np.nan, units="A")
        # self.add_input("data:mission:sizing:main_route:cruise:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:main_route:descent:battery_current", np.nan, units="A")
        self.add_input("data:mission:sizing:taxi_in:battery_current", np.nan, units="A")

        self.add_input("data:mission:sizing:taxi_out:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:takeoff:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:initial_climb:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:main_route:climb:battery_capacity", np.nan, units="A*h")
        # self.add_input("data:mission:sizing:main_route:cruise:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:main_route:reserve:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:main_route:descent:battery_capacity", np.nan, units="A*h")
        self.add_input("data:mission:sizing:taxi_in:battery_capacity", np.nan, units="A*h")

        self.add_input("data:mission:sizing:taxi_out:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:takeoff:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:initial_climb:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:main_route:climb:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:main_route:cruise:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:main_route:reserve:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:main_route:descent:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:taxi_in:battery_energy", np.nan, units="kW*h")
        self.add_input("settings:electrical_system:SOC_in_reserve", 1, units=None)
        self.add_input("data:mission:sizing:end_of_mission:SOC", 0.2, units=None)

        self.add_output("data:mission:sizing:fuel", val=0.0, units="kg")
        self.add_output("data:mission:sizing:battery_max_current", units="A")
        self.add_output("data:mission:sizing:battery_min_current", units="A")
        self.add_output("data:mission:sizing:total_battery_capacity", units="A*h")
        self.add_output("data:mission:sizing:total_battery_energy", 20, units="kW*h")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Hydrogen mass
        m_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        m_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        m_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]
        m_climb = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cruise = inputs["data:mission:sizing:main_route:cruise:fuel"]
        m_reserve = inputs["data:mission:sizing:main_route:reserve:fuel"]
        # m_descent = inputs["data:mission:sizing:main_route:descent:fuel"]
        m_taxi_in = inputs["data:mission:sizing:taxi_in:fuel"]

        m_total = (
                m_taxi_out
                + m_takeoff
                + m_initial_climb
                + m_climb
                + m_cruise
                + m_reserve
                # + m_descent
                + m_taxi_in
        )

        outputs["data:mission:sizing:fuel"] = m_total

        # Battery energy and capacity
        current_taxi_out = inputs["data:mission:sizing:taxi_out:battery_current"]
        current_takeoff = inputs["data:mission:sizing:takeoff:battery_current"]
        current_initial_climb = inputs["data:mission:sizing:initial_climb:battery_current"]
        current_climb = inputs["data:mission:sizing:main_route:climb:battery_current"]
        # current_cruise = inputs["data:mission:sizing:main_route:cruise:battery_current"]
        current_descent = inputs["data:mission:sizing:main_route:descent:battery_current"]
        current_taxi_in = inputs["data:mission:sizing:taxi_in:battery_current"]

        capacity_taxi_out = inputs["data:mission:sizing:taxi_out:battery_capacity"]
        capacity_takeoff = inputs["data:mission:sizing:takeoff:battery_capacity"]
        capacity_initial_climb = inputs["data:mission:sizing:initial_climb:battery_capacity"]
        capacity_climb = inputs["data:mission:sizing:main_route:climb:battery_capacity"]
        # capacity_cruise = inputs["data:mission:sizing:main_route:cruise:battery_capacity"]
        capacity_reserve = inputs["data:mission:sizing:main_route:reserve:battery_capacity"]
        capacity_descent = inputs["data:mission:sizing:main_route:descent:battery_capacity"]
        capacity_taxi_in = inputs["data:mission:sizing:taxi_in:battery_capacity"]

        energy_taxi_out = inputs["data:mission:sizing:taxi_out:battery_energy"]
        energy_takeoff = inputs["data:mission:sizing:takeoff:battery_energy"]
        energy_initial_climb = inputs["data:mission:sizing:initial_climb:battery_energy"]
        energy_climb = inputs["data:mission:sizing:main_route:climb:battery_energy"]
        # energy_cruise = inputs["data:mission:sizing:main_route:cruise:battery_energy"]
        # energy_reserve = inputs["data:mission:sizing:main_route:reserve:battery_energy"]
        energy_descent = inputs["data:mission:sizing:main_route:descent:battery_energy"]
        energy_taxi_in = inputs["data:mission:sizing:taxi_in:battery_energy"]

        SOC_choice = inputs["settings:electrical_system:SOC_in_reserve"]
        SOC_remaining = inputs["data:mission:sizing:end_of_mission:SOC"]

        energy_total = (
                energy_taxi_out
                + energy_takeoff
                + energy_initial_climb
                + energy_climb
                # + energy_cruise
                # + energy_reserve
                + energy_descent
                + energy_taxi_in
        )

        capacity_total = (
                capacity_taxi_out
                + capacity_takeoff
                + capacity_initial_climb
                + capacity_climb
                # + capacity_cruise
                + capacity_reserve
                + capacity_descent
                + capacity_taxi_in
        )

        # Battery max current
        max_current = max(
            current_taxi_out,
            current_takeoff,
            current_initial_climb,
            current_climb,
            # current_cruise,
            current_descent,
            current_taxi_in
        )
        min_current = min(
            current_taxi_out,
            current_takeoff,
            current_initial_climb,
            current_climb,
            # current_cruise,
            current_descent,
            current_taxi_in
        )

        outputs["data:mission:sizing:battery_max_current"] = max_current
        outputs["data:mission:sizing:battery_min_current"] = min_current
        outputs["data:mission:sizing:total_battery_capacity"] = capacity_total

        # Battery energy computation based on the design choice for the user
        # Choice 1 --> x% battery energy to be remaining at the end of mission for battery safety is included within
        # the reserve phase
        # Choice 0 --> x% battery energy to be remaining at the end of mission for battery safety is not included
        # within the reserve phase

        if SOC_choice == 1:
            outputs["data:mission:sizing:total_battery_energy"] = energy_total
        else:
            outputs["data:mission:sizing:total_battery_energy"] = (1 + SOC_remaining) * energy_total


class _Atmosphere(Atmosphere):
    def __init__(
            self,
            altitude: Union[float, Sequence[float]],
            delta_t: float = 0.0,
            altitude_in_feet: bool = True,
    ):
        super().__init__(altitude, delta_t, altitude_in_feet)
        self._calibrated_airspeed = None

    @property
    def true_airspeed(self) -> Union[float, Sequence[float]]:
        """True airspeed (TAS) in m/s."""
        # Dev note: true_airspeed is the "hub". Other speed values will be calculated
        # from this true_airspeed.
        if self._true_airspeed is None:
            if self._mach is not None:
                self._true_airspeed = self._mach * self.speed_of_sound
            if self._equivalent_airspeed is not None:
                sea_level = Atmosphere(0)
                self._true_airspeed = self._return_value(
                    self._equivalent_airspeed * np.sqrt(sea_level.density / self.density)
                )
            if self._unitary_reynolds is not None:
                self._true_airspeed = self._unitary_reynolds * self.kinematic_viscosity
            if self._calibrated_airspeed is not None:
                sea_level = Atmosphere(0)
                current_level = Atmosphere(self._altitude, altitude_in_feet=False)
                impact_pressure = sea_level.pressure * (
                        (
                                (np.asarray(self._calibrated_airspeed) / sea_level.speed_of_sound) ** 2.0
                                / 5.0
                                + 1.0
                        )
                        ** 3.5
                        - 1.0
                )
                total_pressure = current_level.pressure + impact_pressure
                sigma_0 = total_pressure / current_level.pressure
                gamma = 1.4
                mach = (2.0 / (gamma - 1.0) * (sigma_0 ** ((gamma - 1.0) / gamma) - 1.0)) ** 0.5
                self._true_airspeed = mach * current_level.speed_of_sound
        return self._return_value(self._true_airspeed)

    @property
    def calibrated_airspeed(self) -> Union[float, Sequence[float]]:
        """Calibrated airspeed (CAS) in m/s."""
        if self._calibrated_airspeed is None:
            if self._true_airspeed is not None:
                sea_level = Atmosphere(0)
                current_level = Atmosphere(self._altitude, altitude_in_feet=False)
                mach = np.asarray(self._true_airspeed) / current_level.speed_of_sound
                gamma = 1.4
                sigma_0 = (1.0 + (gamma - 1.0) / 2.0 * mach ** 2.0) ** (gamma / (gamma - 1.0))
                total_pressure = sigma_0 * current_level.pressure
                impact_pressure = total_pressure - current_level.pressure
                self._calibrated_airspeed = (
                        sea_level.speed_of_sound
                        * (5.0 * ((impact_pressure / sea_level.pressure + 1.0) ** (1.0 / 3.5) - 1.0))
                        ** 0.5
                )
            if self._mach is not None:
                sea_level = Atmosphere(0)
                current_level = Atmosphere(self._altitude, altitude_in_feet=False)
                gamma = 1.4
                sigma_0 = (1.0 + (gamma - 1.0) / 2.0 * self._mach ** 2.0) ** (gamma / (gamma - 1.0))
                total_pressure = sigma_0 * current_level.pressure
                impact_pressure = total_pressure - current_level.pressure
                self._calibrated_airspeed = (
                        sea_level.speed_of_sound
                        * (5.0 * ((impact_pressure / sea_level.pressure + 1.0) ** (1.0 / 3.5) - 1.0))
                        ** 0.5
                )
        return self._return_value(self._calibrated_airspeed)

    @true_airspeed.setter
    def true_airspeed(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        self._true_airspeed = value

    @calibrated_airspeed.setter
    def calibrated_airspeed(self, value: Union[float, Sequence[float]]):
        self._reset_speeds()
        self._calibrated_airspeed = value

    def _reset_speeds(self):
        """To be used before setting a new speed value as private attribute."""
        self._mach = None
        self._true_airspeed = None
        self._calibrated_airspeed = None
        self._equivalent_airspeed = None
        self._unitary_reynolds = None


class _compute_taxi(om.ExplicitComponent):
    """
    Compute the fuel consumption and battery power and capacity for taxi based on speed and duration.
    Since no 'TAXI' EngineSetting has been implemented IDLE setting is chosen for now
    (meaning fuel cell nos used in this phase : maybe chose cruise phase instead).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("taxi_out", default=True, types=bool)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("settings:electrical_system:system_voltage", np.nan, units="V")

        if self.options["taxi_out"]:
            self.add_input("data:mission:sizing:taxi_out:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
            self.add_output("data:mission:sizing:taxi_out:fuel", units="kg")
            self.add_output("data:mission:sizing:taxi_out:battery_power", units='W')
            self.add_output("data:mission:sizing:taxi_out:battery_capacity", units='A*h')
            self.add_output("data:mission:sizing:taxi_out:battery_current", units='A')
            self.add_output("data:mission:sizing:taxi_out:battery_energy", units='kW*h')
        else:
            self.add_input("data:mission:sizing:taxi_in:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_in:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
            self.add_output("data:mission:sizing:taxi_in:fuel", units="kg")
            self.add_output("data:mission:sizing:taxi_in:battery_power", units='W')
            self.add_output("data:mission:sizing:taxi_in:battery_capacity", units='A*h')
            self.add_output("data:mission:sizing:taxi_in:battery_current", units='A')
            self.add_output("data:mission:sizing:taxi_in:battery_energy", units='kW*h')

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["taxi_out"]:
            _LOGGER.info("Entering mission computation")

        propulsion_model = HybridEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        if self.options["taxi_out"]:
            thrust_rate = inputs["data:mission:sizing:taxi_out:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_out:duration"]
            mach = inputs["data:mission:sizing:taxi_out:speed"] / Atmosphere(0.0).speed_of_sound
        else:
            thrust_rate = inputs["data:mission:sizing:taxi_in:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_in:duration"]
            mach = inputs["data:mission:sizing:taxi_in:speed"] / Atmosphere(0.0).speed_of_sound

        # FIXME: no specific settings for taxi (to be changed in fastoad\constants.py)
        flight_point = FlightPoint(
            mach=mach, altitude=0.0, engine_setting=EngineSetting.IDLE, thrust_rate=thrust_rate
        )
        flight_point.add_field("battery_power", annotation_type=float)
        propulsion_model.compute_flight_points(flight_point)
        hyd_mass = propulsion_model.get_consumed_mass(flight_point, duration)

        if self.options["taxi_out"]:
            outputs["data:mission:sizing:taxi_out:fuel"] = hyd_mass
        else:
            outputs["data:mission:sizing:taxi_in:fuel"] = hyd_mass

            # The electrical system voltage is used to compute the current
            system_voltage = inputs["settings:electrical_system:system_voltage"]

            # Compute the engine power during taxi and subsequently, the current, capacity and energy
            taxi_power = flight_point.battery_power
            battery_current = taxi_power / system_voltage
            taxi_bat_capacity = battery_current * duration / 3600
            bat_energy_taxi_out = propulsion_model.get_consumed_energy(flight_point, duration / 3600) / 1000  # kWh

            if self.options["taxi_out"]:
                outputs["data:mission:sizing:taxi_out:battery_power"] = taxi_power
                outputs["data:mission:sizing:taxi_out:battery_current"] = battery_current
                outputs["data:mission:sizing:taxi_out:battery_capacity"] = taxi_bat_capacity
                outputs["data:mission:sizing:taxi_out:battery_energy"] = bat_energy_taxi_out
            else:
                outputs["data:mission:sizing:taxi_in:battery_power"] = taxi_power
                outputs["data:mission:sizing:taxi_in:battery_current"] = battery_current
                outputs["data:mission:sizing:taxi_in:battery_capacity"] = taxi_bat_capacity
                outputs["data:mission:sizing:taxi_in:battery_energy"] = bat_energy_taxi_out


class _compute_climb(DynamicEquilibrium):
    """
    Compute the hydrogen consumption and the battery energy on climb segment with constant VCAS and fixed thrust ratio.
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

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:takeoff:battery_energy", np.nan, units="kW*h")
        self.add_input("data:mission:sizing:initial_climb:battery_energy", np.nan, units="kW*h")
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:sea_level", val=np.nan, units="m/s"
        )
        self.add_input(
            "data:mission:sizing:main_route:climb:climb_rate:cruise_level", val=np.nan, units="m/s"
        )
        self.add_input("settings:electrical_system:system_voltage", np.nan, units="V")

        self.add_output("data:mission:sizing:main_route:climb:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:climb:battery_capacity", units='A*h')
        self.add_output("data:mission:sizing:main_route:climb:battery_current", units='A')
        self.add_output("data:mission:sizing:main_route:climb:battery_power", units="W")
        self.add_output("data:mission:sizing:main_route:climb:battery_energy", units="kW*h")
        self.add_output("data:mission:sizing:main_route:climb:distance", units="m")
        self.add_output("data:mission:sizing:main_route:climb:duration", units="s")
        self.add_output("data:mission:sizing:main_route:climb:v_cas", units="m/s")
        self.add_output("data:mission:sizing:main_route:climb:battery_power_array", shape=POINTS_POWER_COUNT, units="W")
        self.add_output("data:mission:sizing:main_route:climb:battery_time_array", shape=POINTS_POWER_COUNT, units="h")
        self.add_output("data:mission:sizing:main_route:climb:battery_capacity_array", shape=POINTS_POWER_COUNT, units="A*h")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Delete previous .csv results
        if self.options["out_file"] != "":
            # noinspection PyBroadException
            try:
                os.remove(self.options["out_file"])
            except:
                _LOGGER.info("Failed to remove {} file!".format(self.options["out_file"]))

        propulsion_model = HybridEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        climb_rate_sl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:sea_level"])
        climb_rate_cl = float(inputs["data:mission:sizing:main_route:climb:climb_rate:cruise_level"])
        system_voltage = inputs["settings:electrical_system:system_voltage"]

        # Define initial conditions of the hydrogen fuel cells system
        t_start = time.time()
        altitude_t = SAFETY_HEIGHT  # conversion to m
        distance_t = 0.0
        time_t = 0.0
        mass_t = mtow - (m_to + m_tk + m_ic)
        mass_fuel_t = 0.0
        previous_step = ()

        # Define initial conditions of the battery(ies)
        bat_capacity = 0.0
        climb_current = [0]  # Array used to get the maximum value of current
        current_climb = 0.0
        bat_energy_climb = 0.0
        climb_power = [0]  # Array used to get the maximum value of power
        climb_time = [0]
        climb_capacity = [0]
        power_climb = 0.0
        atm_0 = Atmosphere(0.0)
        previous_step = ()

        # Calculate constant speed (cos(gamma)~1) and corresponding climb angle
        # FIXME: VCAS constant-speed strategy is specific to ICE-propeller configuration, should be an input!
        cl = math.sqrt(3 * cd0 / coef_k_wing)
        atm = _Atmosphere(altitude_t, altitude_in_feet=False)
        vs1 = math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl_max_clean))
        v_cas = max(math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl)), 1.3 * vs1)
        atm.calibrated_airspeed = v_cas
        v_tas = atm.true_airspeed
        gamma = math.asin(climb_rate_sl / v_tas)

        # Define specific time step ~POINTS_NB_CLIMB points for calculation (with ground conditions)
        time_step = ((cruise_altitude - SAFETY_HEIGHT) / climb_rate_sl) / float(POINTS_NB_CLIMB)

        while altitude_t < cruise_altitude:

            # Calculate dynamic pressure
            atm = _Atmosphere(altitude_t, altitude_in_feet=False)
            atm.calibrated_airspeed = v_cas
            v_tas = atm.true_airspeed
            climb_rate = interp1d([0.0, float(cruise_altitude)], [climb_rate_sl, climb_rate_cl])(
                altitude_t
            )
            gamma = math.asin(climb_rate / v_tas)
            mach = v_tas / atm.speed_of_sound
            atm_1 = _Atmosphere(altitude_t + 1.0, altitude_in_feet=False)
            atm_1.calibrated_airspeed = v_cas
            dv_tas_dh = atm_1.true_airspeed - v_tas
            dvx_dt = dv_tas_dh * v_tas * math.sin(gamma)
            q = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium
            previous_step = self.dynamic_equilibrium(
                inputs, gamma, q, dvx_dt, 0.0, mass_t, "none", previous_step[0:2]
            )
            thrust = float(previous_step[1])

            # Compute consumption
            flight_point = FlightPoint(
                mach=mach,
                altitude=altitude_t,
                engine_setting=EngineSetting.CLIMB,
                thrust_is_regulated=True,
                thrust=thrust,
            )
            flight_point.add_field("battery_power", annotation_type=float)
            propulsion_model.compute_flight_points(flight_point)
            if flight_point.thrust_rate > 1.0:
                _LOGGER.warning("Thrust rate is above 1.0, value clipped at 1.0")

            # Save results
            if self.options["out_file"] != "":
                self.save_point(
                    time_t,
                    altitude_t,
                    distance_t,
                    mass_t,
                    v_tas,
                    v_cas,
                    atm.density,
                    gamma * 180.0 / math.pi,
                    previous_step,
                    flight_point.thrust_rate,
                    flight_point.sfc,
                    "sizing:main_route:climb",
                )

            consumed_mass_1s = propulsion_model.get_consumed_mass(flight_point, 1.0)

            # Calculate distance variation (earth axis)
            v_z = v_tas * math.sin(gamma)
            v_x = v_tas * math.cos(gamma)
            time_step = min(time_step, (cruise_altitude - altitude_t) / v_z)
            altitude_t += v_z * time_step
            distance_t += v_x * time_step

            # Estimate mass evolution and update time
            mass_fuel_t += consumed_mass_1s * time_step
            mass_t = mass_t - consumed_mass_1s * time_step
            time_t += time_step

            # Estimate battery energy and update time
            climb_power.append(flight_point.battery_power)

            power_climb = max(climb_power)
            climb_current.append(flight_point.battery_power / system_voltage)
            current_climb = max(climb_current)

            # Since the time step is in seconds and the energy should be computed in kWh, time step is divided by 3600
            bat_capacity += (flight_point.battery_power / system_voltage) * time_step / 3600
            climb_capacity.append((flight_point.battery_power / system_voltage) * time_step / 3600)
            bat_energy_climb += propulsion_model.get_consumed_energy(flight_point, time_step / 3600) / 1000  # [kWh]
            # time_t += time_step

            climb_time.append(time_t / 3600)

            # Check calculation duration
            if (time.time() - t_start) > MAX_CALCULATION_TIME:
                raise Exception(
                    "Time calculation duration for climb phase [{}s] exceeded!".format(
                        MAX_CALCULATION_TIME
                    )
                )

        # Add additional zeros in the power array to meet the plot requirements during post-processing
        while len(climb_power) < POINTS_POWER_COUNT:
            climb_power.append(0)
            climb_time.append(0)
            climb_capacity.append(0)

        # Save results
        if self.options["out_file"] != "":
            self.save_point(
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
            )

        outputs["data:mission:sizing:main_route:climb:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:climb:distance"] = distance_t
        outputs["data:mission:sizing:main_route:climb:duration"] = time_t
        outputs["data:mission:sizing:main_route:climb:v_cas"] = v_cas
        outputs["data:mission:sizing:main_route:climb:battery_power"] = power_climb
        outputs["data:mission:sizing:main_route:climb:battery_current"] = current_climb
        outputs["data:mission:sizing:main_route:climb:battery_capacity"] = bat_capacity
        outputs["data:mission:sizing:main_route:climb:battery_energy"] = bat_energy_climb
        outputs["data:mission:sizing:main_route:climb:battery_power_array"] = climb_power
        outputs["data:mission:sizing:main_route:climb:battery_time_array"] = climb_time
        outputs["data:mission:sizing:main_route:climb:battery_capacity_array"] = climb_capacity


class _compute_cruise(DynamicEquilibrium):
    """
    Compute the hydrogen consumption on cruise segment with constant VTAS and altitude.
    Assumption is made that we rely solely on the fuel cell system's power during cruise phase.
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

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:TLAR:range", np.nan, units="m")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        # self.add_input("data:mission:sizing:taxi_out:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:holding:battery_energy", 0.0, units="kW*h")
        # self.add_input("data:mission:sizing:takeoff:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:initial_climb:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:main_route:climb:battery_energy", np.nan, units="kW*h")
        # self.add_input("settings:electrical_system:system_voltage", np.nan, units="V")
        self.add_input("data:mission:sizing:main_route:climb:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:descent:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:climb:duration", np.nan, units="s")

        self.add_output("data:mission:sizing:main_route:cruise:fuel", units="kg")
        # self.add_output("data:mission:sizing:main_route:cruise:battery_capacity", units='A*h')
        # self.add_output("data:mission:sizing:main_route:cruise:battery_current", units='A')
        # self.add_output("data:mission:sizing:main_route:cruise:battery_power", units="W")
        # self.add_output("data:mission:sizing:main_route:cruise:battery_energy", units="kW*h")
        # self.add_output("data:mission:sizing:main_route:cruise:battery_power_array", shape=POINTS_POWER_COUNT, units="W")
        # self.add_output("data:mission:sizing:main_route:cruise:battery_time_array", shape=POINTS_POWER_COUNT, units="h")
        # self.add_output("data:mission:sizing:main_route:cruise:battery_capacity_array", shape=POINTS_POWER_COUNT, units="A*h")
        self.add_output("data:mission:sizing:main_route:cruise:distance", units="m")
        self.add_output("data:mission:sizing:main_route:cruise:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = HybridEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        v_tas = inputs["data:TLAR:v_cruise"]
        cruise_distance = max(
            0.0,
            (
                    inputs["data:TLAR:range"]
                    - inputs["data:mission:sizing:main_route:climb:distance"]
                    - inputs["data:mission:sizing:main_route:descent:distance"]
            ),
        )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]
        # system_voltage = inputs["settings:electrical_system:system_voltage"]

        # Define specific time step ~POINTS_NB_CRUISE points for calculation
        time_step = (cruise_distance / v_tas) / float(POINTS_NB_CRUISE)

        # Define initial conditions
        t_start = time.time()
        distance_t = 0.0
        time_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_tk + m_ic + m_cl)
        # cruise_power = [0]  # Array used to get the maximum value of power at the end
        cruise_time = [0]
        # cruise_capacity = [0]
        # power_cruise = 0.0
        # cruise_current = [0]  # Array used to get the maximum value of current at the end
        # current_cruise = 0.0
        # bat_capacity_cruise = 0.0
        # bat_energy_cruise = 0.0
        atm = _Atmosphere(cruise_altitude, altitude_in_feet=False)
        atm.true_airspeed = v_tas
        mach = atm.mach
        previous_step = ()

        while distance_t < cruise_distance:

            # Calculate dynamic pressure
            q = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium
            previous_step = self.dynamic_equilibrium(
                inputs, 0.0, q, 0.0, 0.0, mass_t, "none", previous_step[0:2]
            )
            thrust = float(previous_step[1])

            # Compute consumption
            flight_point = FlightPoint(
                mach=mach,
                altitude=cruise_altitude,
                engine_setting=EngineSetting.CRUISE,
                thrust_is_regulated=True,
                thrust=thrust,
            )
            flight_point.add_field("battery_power", annotation_type=float)
            propulsion_model.compute_flight_points(flight_point)
            if flight_point.thrust_rate > 1.0:
                _LOGGER.warning("Thrust rate is above 1.0, value clipped at 1.0")

            # Save results
            if self.options["out_file"] != "":
                self.save_point(
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

            # # Estimate the battery energy consumption, capacity, current and update cruise duration
            # cruise_power.append(flight_point.battery_power)
            # power_cruise = max(cruise_power)
            #
            # cruise_current.append(flight_point.battery_power / system_voltage)
            # current_cruise = max(cruise_current)
            #
            # # Time step is divided by 3600 to compute the capacity in A*h and energy in kWh
            # bat_capacity_cruise += (flight_point.battery_power / system_voltage) * time_step / 3600
            # cruise_capacity.append((flight_point.battery_power / system_voltage) * time_step / 3600)
            #
            # bat_energy_cruise += propulsion_model.get_consumed_energy(
            #     flight_point,
            #     min(time_step / 3600, (cruise_distance - distance_t) / v_tas)
            # )
            # time_t += min(time_step, (cruise_distance - distance_t) / v_tas)
            # cruise_time.append(time_t / 3600)

            # Check calculation duration
            if (time.time() - t_start) > MAX_CALCULATION_TIME:
                raise Exception(
                    "Time calculation duration for cruise phase [{}s] exceeded!".format(
                        MAX_CALCULATION_TIME
                    )
                )

        # Add additional zeros in the power array to meet the plot requirements during post-processing
        # while len(cruise_power) < POINTS_POWER_COUNT:
        #     cruise_power.append(0)
        #     cruise_time.append(0)
        #     cruise_capacity.append(0)

        # Save results
        if self.options["out_file"] != "":
            self.save_point(
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
            )

        outputs["data:mission:sizing:main_route:cruise:fuel"] = mass_fuel_t
        # outputs["data:mission:sizing:main_route:cruise:power"] = power_cruise
        # outputs["data:mission:sizing:main_route:cruise:current"] = current_cruise
        # outputs["data:mission:sizing:main_route:cruise:battery_capacity"] = bat_capacity_cruise
        # outputs["data:mission:sizing:main_route:cruise:battery_energy"] = bat_energy_cruise
        # outputs["data:mission:sizing:main_route:cruise:battery_power_array"] = cruise_power
        # outputs["data:mission:sizing:main_route:cruise:battery_time_array"] = cruise_time
        # outputs["data:mission:sizing:main_route:cruise:battery_capacity_array"] = cruise_capacity
        outputs["data:mission:sizing:main_route:cruise:distance"] = distance_t
        outputs["data:mission:sizing:main_route:cruise:duration"] = time_t


class _compute_descent(DynamicEquilibrium):
    """
    Compute the battery energy on descent segment with constant VCAS and descent rate.
    The hypothesis of small alpha angle is done.
    Warning: Descent rate is reduced if cd/cl < abs(desc_rate)!
    Assumption is made that we rely solely on battery power during descent and that fuel cells are deactivated.
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

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:mission:sizing:main_route:descent:descent_rate", np.nan, units="m/s")
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        # self.add_input("data:mission:sizing:taxi_out:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:holding:battery_energy", 0.0, units="kW*h")
        # self.add_input("data:mission:sizing:takeoff:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:initial_climb:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:main_route:climb:battery_energy", np.nan, units="kW*h")
        # self.add_input("data:mission:sizing:main_route:cruise:battery_energy", np.nan, units="kW*h")
        self.add_input("settings:electrical_system:system_voltage", np.nan, units="V")
        self.add_input("data:mission:sizing:main_route:climb:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:climb:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:cruise:duration", np.nan, units="s")

        # self.add_output("data:mission:sizing:main_route:descent:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:descent:battery_capacity", units='A*h')
        self.add_output("data:mission:sizing:main_route:descent:battery_current", units='A')
        self.add_output("data:mission:sizing:main_route:descent:battery_power", units="W")
        self.add_output("data:mission:sizing:main_route:descent:battery_energy", units="kW*h")
        self.add_output("data:mission:sizing:main_route:descent:battery_power_array", shape=POINTS_POWER_COUNT, units="W")
        self.add_output("data:mission:sizing:main_route:descent:battery_time_array", shape=POINTS_POWER_COUNT, units="h")
        self.add_output("data:mission:sizing:main_route:descent:battery_capacity_array", shape=POINTS_POWER_COUNT, units="A*h")
        self.add_output("data:mission:sizing:main_route:descent:distance", 0.0, units="m")
        self.add_output("data:mission:sizing:main_route:descent:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = HybridEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        descent_rate = inputs["data:mission:sizing:main_route:descent:descent_rate"]
        cl = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cr = inputs["data:mission:sizing:main_route:cruise:fuel"]
        system_voltage = inputs["settings:electrical_system:system_voltage"]

        # Define initial conditions
        t_start = time.time()
        altitude_t = copy.deepcopy(cruise_altitude)
        distance_t = 0.0
        time_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_tk + m_ic + m_cl + m_cr)
        previous_step = ()

        descent_power = [0]
        descent_time = [0]
        descent_capacity = [0]
        power_descent = 0.0
        descent_current = [0]
        current_descent = 0.0
        bat_capacity_descent = 0.0
        bat_energy_descent = 0.0
        # atm_0 = Atmosphere(0.0)
        # warning = False

        # Calculate constant speed (cos(gamma)~1) and corresponding descent angle
        # FIXME: VCAS constant-speed strategy is specific to ICE-propeller configuration, should be an input!
        atm = _Atmosphere(altitude_t)
        vs1 = math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl_max_clean))
        v_cas = max(math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl)), 1.3 * vs1)
        atm.calibrated_airspeed = v_cas
        v_tas = atm.true_airspeed
        gamma = math.asin(descent_rate / v_cas)

        # Define specific time step ~POINTS_NB_CLIMB points for calculation (with ground conditions)
        time_step = abs((altitude_t / descent_rate)) / float(POINTS_NB_DESCENT)

        while altitude_t > 0.0:

            # Calculate dynamic pressure
            atm = _Atmosphere(altitude_t, altitude_in_feet=False)
            atm.calibrated_airspeed = v_cas
            v_tas = atm.true_airspeed
            mach = v_tas / atm.speed_of_sound
            atm_1 = _Atmosphere(altitude_t + 1.0, altitude_in_feet=False)
            atm_1.calibrated_airspeed = v_cas
            dv_tas_dh = atm_1.true_airspeed - v_tas
            dvx_dt = dv_tas_dh * v_tas * math.sin(gamma)
            q = 0.5 * atm.density * v_tas ** 2

            # Find equilibrium, decrease gamma if obtained thrust is negative
            previous_step = self.dynamic_equilibrium(
                inputs, gamma, q, dvx_dt, 0.0, mass_t, "none", previous_step[0:2]
            )
            thrust = previous_step[1]
            while thrust < 0.0:
                gamma = 0.9 * gamma
                previous_step = self.dynamic_equilibrium(
                    inputs, gamma, q, dvx_dt, 0.0, mass_t, "none", previous_step[0:2]
                )
                thrust = previous_step[1]

            # Compute consumption
            # FIXME: DESCENT setting on engine does not exist, replaced by IDLE for test
            flight_point = FlightPoint(
                mach=mach,
                altitude=altitude_t,
                engine_setting=EngineSetting.IDLE,
                thrust_is_regulated=True,
                thrust=thrust,
            )
            flight_point.add_field("battery_power", annotation_type=float)
            propulsion_model.compute_flight_points(flight_point)
            # Save results
            if self.options["out_file"] != "":
                self.save_point(
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
                )
            consumed_mass_1s = propulsion_model.get_consumed_mass(flight_point, 1.0)

            # Calculate distance variation (earth axis)
            v_x = v_tas * math.cos(gamma)
            v_z = v_tas * math.sin(gamma)
            time_step = min(time_step, -altitude_t / v_z)
            distance_t += v_x * time_step
            altitude_t += v_z * time_step

            # Estimate mass evolution and update time - no variation in descent phase under the assumption of battery
            # power only
            # mass_fuel_t += consumed_mass_1s * time_step
            # mass_t = mass_t - consumed_mass_1s *
            # time_step time_t += time_step

            # Estimate battery energy consumption, capacity, current and update descent duration
            descent_power.append(flight_point.battery_power)
            power_descent = max(descent_power)

            descent_current.append(flight_point.battery_power / system_voltage)
            current_descent = max(descent_current)
            bat_capacity_descent += (flight_point.battery_power / system_voltage) * time_step / 3600  # [Ah]
            descent_capacity.append((flight_point.battery_power / system_voltage) * time_step / 3600)
            bat_energy_descent += propulsion_model.get_consumed_energy(flight_point, time_step / 3600) / 1000  # [kWh]
            # Time step is divided by 3600 to compute the energy in kWh
            time_t += time_step
            descent_time.append(time_t / 3600)

            # Check calculation duration
            if (time.time() - t_start) > MAX_CALCULATION_TIME:
                raise Exception(
                    "Time calculation duration for descent phase [{}s] exceeded!".format(
                        MAX_CALCULATION_TIME
                    )
                )

        # Add additional zeros in the power array to meet the plot requirements during post-processing
        while len(descent_power) < POINTS_POWER_COUNT:
            descent_power.append(0)
            descent_time.append(0)
            descent_capacity.append(0)

        # Save results
        if self.options["out_file"] != "":
            self.save_point(
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
            )

        # outputs["data:mission:sizing:main_route:descent:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:descent:battery_power"] = power_descent
        outputs["data:mission:sizing:main_route:descent:battery_current"] = current_descent
        outputs["data:mission:sizing:main_route:descent:battery_capacity"] = bat_capacity_descent
        outputs["data:mission:sizing:main_route:descent:battery_energy"] = bat_energy_descent
        outputs["data:mission:sizing:main_route:descent:battery_power_array"] = descent_power
        outputs["data:mission:sizing:main_route:descent:battery_time_array"] = descent_time
        outputs["data:mission:sizing:main_route:descent:battery_capacity_array"] = descent_capacity
        outputs["data:mission:sizing:main_route:descent:distance"] = distance_t
        outputs["data:mission:sizing:main_route:descent:duration"] = time_t
