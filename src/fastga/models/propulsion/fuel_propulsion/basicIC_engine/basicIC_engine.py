"""Parametric propeller IC engine."""
# -*- coding: utf-8 -*-
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
import pandas as pd
from typing import Union, Sequence, Tuple, Optional
from scipy.interpolate import interp2d
import os.path as pth
import numpy as np

import fastoad.api as oad
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError
from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes

from .exceptions import FastBasicICEngineInconsistentInputParametersError
from . import resources

# Logger for this module
_LOGGER = logging.getLogger(__name__)

# Set of dictionary keys that are mapped to instance attributes.
ENGINE_LABELS = {
    "power_SL": dict(doc="Power at sea level in watts."),
    "mass": dict(doc="Mass in kilograms."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}
# Set of dictionary keys that are mapped to instance attributes.
NACELLE_LABELS = {
    "wet_area": dict(doc="Wet area in meters²."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}


class BasicICEngine(AbstractFuelPropulsion):
    def __init__(
        self,
        max_power: float,
        cruise_altitude_propeller: float,
        fuel_type: float,
        strokes_nb: float,
        prop_layout: float,
        k_factor_sfc: float,
        speed_SL,
        thrust_SL,
        thrust_limit_SL,
        efficiency_SL,
        speed_CL,
        thrust_CL,
        thrust_limit_CL,
        efficiency_CL,
        effective_J,
        effective_efficiency_ls,
        effective_efficiency_cruise,
    ):
        """
        Parametric Internal Combustion engine.

        It computes engine characteristics using fuel type, motor architecture
        and constant propeller efficiency using analytical model from following sources:

        :param max_power: maximum delivered mechanical power of engine (units=W)
        :param cruise_altitude_propeller: design altitude for cruise (units=m)
        :param cruise_speed: design altitude for cruise (units=m/s)
        :param fuel_type: 1.0 for gasoline and 2.0 for diesel engine and 3.0 for Jet Fuel
        :param strokes_nb: can be either 2-strokes (=2.0) or 4-strokes (=4.0)
        :param prop_layout: propulsion position in nose (=3.0) or wing (=1.0)
        """
        if fuel_type == 1.0:
            self.ref = {
                "max_power": 132480,
                "length": 0.83,
                "height": 0.57,
                "width": 0.85,
                "mass": 136,
            }  # Lycoming IO-360-B1A
            self.map_file_path = pth.join(resources.__path__[0], "FourCylindersAtmospheric.csv")
        else:
            self.ref = {
                "max_power": 160000,
                "length": 0.859,
                "height": 0.659,
                "width": 0.650,
                "mass": 205,
            }  # TDA CR 1.9 16V
            # FIXME: change the map file for those engines
            self.map_file_path = pth.join(resources.__path__[0], "FourCylindersAtmospheric.csv")
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.cruise_altitude_propeller = cruise_altitude_propeller
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb
        self.idle_thrust_rate = 0.01
        self.k_factor_sfc = k_factor_sfc
        self.speed_SL = speed_SL
        self.thrust_SL = thrust_SL
        self.thrust_limit_SL = thrust_limit_SL
        self.efficiency_SL = efficiency_SL
        self.speed_CL = speed_CL
        self.thrust_CL = thrust_CL
        self.thrust_limit_CL = thrust_limit_CL
        self.efficiency_CL = efficiency_CL
        self.effective_J = float(effective_J)
        self.effective_efficiency_ls = float(effective_efficiency_ls)
        self.effective_efficiency_cruise = float(effective_efficiency_cruise)
        self.specific_shape = None

        # Evaluate engine volume based on max power @ 0.0m
        rpm_vect, pme_vect, pme_limit_vect, sfc_matrix = self.read_map(self.map_file_path)
        self.rpm_vect_ref = rpm_vect
        self.pme_vect_ref = pme_vect
        self.pme_limit_vect_ref = pme_limit_vect
        self.sfc_matrix_ref = sfc_matrix
        volume = self.max_power / np.max(
            pme_limit_vect * 1e5 * rpm_vect / 240.0
        )  # conversion rpm to rad/s included
        self.volume = volume

        # Declare sub-components attribute
        self.engine = Engine(power_SL=max_power)
        self.engine.mass = None
        self.engine.length = None
        self.engine.width = None
        self.engine.height = None

        self.nacelle = Nacelle()
        self.nacelle.wet_area = None

        self.propeller = None

        # This dictionary is expected to have a Mixture coefficient for all EngineSetting values
        self.mixture_values = {
            EngineSetting.TAKEOFF: 1.15,
            EngineSetting.CLIMB: 1.15,
            EngineSetting.CRUISE: 1.0,
            EngineSetting.IDLE: 1.0,
        }
        self.rpm_values = {
            EngineSetting.TAKEOFF: 2700.0,
            EngineSetting.CLIMB: 2700.0,
            EngineSetting.CRUISE: 2500.0,
            EngineSetting.IDLE: 2300.0,
        }

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.mixture_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", str(unknown_keys))

    @staticmethod
    def read_map(map_file_path):

        data = pd.read_csv(map_file_path)
        values = data.to_numpy()[:, 1:].tolist()
        labels = data.to_numpy()[:, 0].tolist()
        data = pd.DataFrame(values, index=labels)
        rpm = data.loc["rpm", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            rpm = rpm.replace("  ", " ")
        rpm_vect = np.array([float(i) for i in rpm.split(" ") if i != ""])
        pme = data.loc["pme", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            pme = pme.replace("  ", " ")
        pme_vect = np.array([float(i) for i in pme.split(" ") if i != ""])
        pme_limit = data.loc["pme_limit", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            pme_limit = pme_limit.replace("  ", " ")
        pme_limit_vect = np.array([float(i) for i in pme_limit.split(" ") if i != ""])
        sfc = data.loc["sfc", 0][1:-2].replace("\n", "").replace("\r", "")
        sfc_lines = sfc[1:-2].split("] [")
        sfc_matrix = np.zeros(
            (len(np.array([i for i in sfc_lines[0].split(" ") if i != ""])), len(sfc_lines))
        )
        for idx in range(len(sfc_lines)):
            sfc_matrix[:, idx] = np.array([i for i in sfc_lines[idx].split(" ") if i != ""])

        return rpm_vect, pme_vect, pme_limit_vect, sfc_matrix

    def compute_flight_points(self, flight_points: oad.FlightPoint):
        # pylint: disable=too-many-arguments
        # they define the trajectory
        self.specific_shape = np.shape(flight_points.mach)
        if isinstance(flight_points.mach, float):
            sfc, thrust_rate, thrust = self._compute_flight_points(
                flight_points.mach,
                flight_points.altitude,
                flight_points.engine_setting,
                flight_points.thrust_is_regulated,
                flight_points.thrust_rate,
                flight_points.thrust,
            )
            flight_points.sfc = sfc
            flight_points.thrust_rate = thrust_rate
            flight_points.thrust = thrust
        else:
            mach = np.asarray(flight_points.mach)
            altitude = np.asarray(flight_points.altitude).flatten()
            engine_setting = np.asarray(flight_points.engine_setting).flatten()
            if flight_points.thrust_is_regulated is None:
                thrust_is_regulated = None
            else:
                thrust_is_regulated = np.asarray(flight_points.thrust_is_regulated).flatten()
            if flight_points.thrust_rate is None:
                thrust_rate = None
            else:
                thrust_rate = np.asarray(flight_points.thrust_rate).flatten()
            if flight_points.thrust is None:
                thrust = None
            else:
                thrust = np.asarray(flight_points.thrust).flatten()
            self.specific_shape = np.shape(mach)
            sfc, thrust_rate, thrust = self._compute_flight_points(
                mach.flatten(),
                altitude,
                engine_setting,
                thrust_is_regulated,
                thrust_rate,
                thrust,
            )
            if len(self.specific_shape) != 1:  # reshape data that is not array form
                # noinspection PyUnresolvedReferences
                flight_points.sfc = sfc.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust_rate = thrust_rate.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust = thrust.reshape(self.specific_shape)
            else:
                flight_points.sfc = sfc
                flight_points.thrust_rate = thrust_rate
                flight_points.thrust = thrust

    def _compute_flight_points(
        self,
        mach: Union[float, Sequence],
        altitude: Union[float, Sequence],
        engine_setting: Union[EngineSetting, Sequence],
        thrust_is_regulated: Optional[Union[bool, Sequence]] = None,
        thrust_rate: Optional[Union[float, Sequence]] = None,
        thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence]]:
        """
        Same as :meth:`compute_flight_points`.

        :param mach: Mach number
        :param altitude: (unit=m) altitude w.r.t. to sea level
        :param engine_setting: define engine settings
        :param thrust_is_regulated: tells if thrust_rate or thrust should be used (works element-
        wise)
        :param thrust_rate: thrust rate (unit=none)
        :param thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)
        """
        # Treat inputs (with check on thrust rate <=1.0)
        if thrust_is_regulated is not None:
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_is_regulated, thrust_rate, thrust = self._check_thrust_inputs(
            thrust_is_regulated, thrust_rate, thrust
        )
        thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_rate = np.asarray(thrust_rate)
        thrust = np.asarray(thrust)

        # Get maximum thrust @ given altitude & mach
        atmosphere = Atmosphere(np.asarray(altitude), altitude_in_feet=False)
        mach = np.asarray(mach) + (np.asarray(mach) == 0) * 1e-12
        atmosphere.mach = mach
        max_thrust = self.max_thrust(np.asarray(engine_setting), atmosphere)

        # We compute thrust values from thrust rates when needed
        idx = np.logical_not(thrust_is_regulated)
        if np.size(max_thrust) == 1:
            maximum_thrust = max_thrust
            out_thrust_rate = thrust_rate
            out_thrust = thrust
        else:
            out_thrust_rate = (
                np.full(np.shape(max_thrust), thrust_rate.item())
                if np.size(thrust_rate) == 1
                else thrust_rate
            )
            out_thrust = (
                np.full(np.shape(max_thrust), thrust.item()) if np.size(thrust) == 1 else thrust
            )
            maximum_thrust = max_thrust[idx]
        if np.any(idx):
            out_thrust[idx] = out_thrust_rate[idx] * maximum_thrust
        if np.any(thrust_is_regulated):
            out_thrust[thrust_is_regulated] = np.minimum(
                out_thrust[thrust_is_regulated], max_thrust[thrust_is_regulated]
            )

        # thrust_rate is obtained from entire thrust vector (could be optimized if needed,
        # as some thrust rates that are computed may have been provided as input)
        out_thrust_rate = out_thrust / max_thrust

        # Now SFC (g/kwh) can be computed and converted to sfc_thrust (kg/N) to match computation
        # from turboshaft
        sfc, mech_power = self.sfc(out_thrust, engine_setting, atmosphere)
        sfc_time = (mech_power * 1e-3) * sfc / 3.6e6  # sfc in kg/s
        sfc_thrust = sfc_time / np.maximum(out_thrust, 1e-6)  # avoid 0 division

        return sfc_thrust, out_thrust_rate, out_thrust

    @staticmethod
    def _check_thrust_inputs(
        thrust_is_regulated: Optional[Union[float, Sequence]],
        thrust_rate: Optional[Union[float, Sequence]],
        thrust: Optional[Union[float, Sequence]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Checks that inputs are consistent and return them in proper shape.
        Some of the inputs can be None, but outputs will be proper numpy arrays.
        :param thrust_is_regulated:
        :param thrust_rate:
        :param thrust:
        :return: the inputs, but transformed in numpy arrays.
        """
        # Ensure they are numpy array
        if thrust_is_regulated is not None:
            # As OpenMDAO may provide floats that could be slightly different
            # from 0. or 1., a rounding operation is needed before converting
            # to booleans
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        if thrust_rate is not None:
            thrust_rate = np.asarray(thrust_rate)
        if thrust is not None:
            thrust = np.asarray(thrust)

        # Check inputs: if use_thrust_rate is None, we will use the provided input between
        # thrust_rate and thrust
        if thrust_is_regulated is None:
            if thrust_rate is not None:
                thrust_is_regulated = False
                thrust = np.empty_like(thrust_rate)
            elif thrust is not None:
                thrust_is_regulated = True
                thrust_rate = np.empty_like(thrust)
            else:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is None, either thrust_rate or thrust should be provided."
                )

        elif np.size(thrust_is_regulated) == 1:
            # Check inputs: if use_thrust_rate is a scalar, the matching input(thrust_rate or
            # thrust) must be provided.
            if thrust_is_regulated:
                if thrust is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is True, thrust should be provided."
                    )
                thrust_rate = np.empty_like(thrust)
            else:
                if thrust_rate is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is False, thrust_rate should be provided."
                    )
                thrust = np.empty_like(thrust_rate)

        else:
            # Check inputs: if use_thrust_rate is not a scalar, both thrust_rate and thrust must be
            # provided and have the same shape as use_thrust_rate
            if thrust_rate is None or thrust is None:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When thrust_is_regulated is a sequence, both thrust_rate and thrust should be "
                    "provided."
                )
            if np.shape(thrust_rate) != np.shape(thrust_is_regulated) or np.shape(
                thrust
            ) != np.shape(thrust_is_regulated):
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should have "
                    "same shape as use_thrust_rate"
                )

        return thrust_is_regulated, thrust_rate, thrust

    def propeller_efficiency(
        self, thrust: Union[float, Sequence[float]], atmosphere: Atmosphere
    ) -> Union[float, Sequence]:
        """
        Compute the propeller efficiency.

        :param thrust: Thrust (in N)
        :param atmosphere: Atmosphere instance at intended altitude
        :return: efficiency
        """
        # Include advance ratio loss in here, we will assume that since we work at constant RPM
        # the change in advance ration is equal to a change in velocity
        installed_airspeed = atmosphere.true_airspeed * self.effective_J

        propeller_efficiency_SL = interp2d(
            self.thrust_SL,
            self.speed_SL,
            self.efficiency_SL * self.effective_efficiency_ls,  # Include the efficiency loss
            # in here
            kind="cubic",
        )
        propeller_efficiency_CL = interp2d(
            self.thrust_CL,
            self.speed_CL,
            self.efficiency_CL * self.effective_efficiency_cruise,  # Include the efficiency loss
            # in here
            kind="cubic",
        )
        if isinstance(atmosphere.true_airspeed, float):
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(installed_airspeed, self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(installed_airspeed, self.speed_CL, self.thrust_limit_CL),
            )
        else:
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(list(installed_airspeed), self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(list(installed_airspeed), self.speed_CL, self.thrust_limit_CL),
            )
        if np.size(thrust) == 1:  # calculate for float
            lower_bound = float(propeller_efficiency_SL(thrust_interp_SL, installed_airspeed))
            upper_bound = float(propeller_efficiency_CL(thrust_interp_CL, installed_airspeed))
            altitude = atmosphere.get_altitude(altitude_in_feet=False)
            propeller_efficiency = np.interp(
                altitude, [0, self.cruise_altitude_propeller], [lower_bound, upper_bound]
            )
        else:  # calculate for array
            propeller_efficiency = np.zeros(np.size(thrust))
            for idx in range(np.size(thrust)):
                lower_bound = propeller_efficiency_SL(
                    thrust_interp_SL[idx], installed_airspeed[idx]
                )
                upper_bound = propeller_efficiency_CL(
                    thrust_interp_CL[idx], installed_airspeed[idx]
                )
                altitude = atmosphere.get_altitude(altitude_in_feet=False)[idx]
                propeller_efficiency[idx] = (
                    lower_bound
                    + (upper_bound - lower_bound)
                    * np.minimum(altitude, self.cruise_altitude_propeller)
                    / self.cruise_altitude_propeller
                )

        return propeller_efficiency

    def compute_max_power(self, flight_points: oad.FlightPoint) -> Union[float, Sequence]:
        """
        Compute the ICE maximum power @ given flight-point.

        :param flight_points: current flight point(s)
        :return: maximum power in kW
        """
        atmosphere = Atmosphere(np.asarray(flight_points.altitude), altitude_in_feet=False)
        sigma = atmosphere.density / Atmosphere(0.0).density
        max_power = (self.max_power / 1e3) * (sigma - (1 - sigma) / 7.55)  # max power in kW

        return max_power

    def sfc(
        self,
        thrust: Union[float, Sequence[float]],
        engine_setting: Union[float, Sequence[float]],
        atmosphere: Atmosphere,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computation of the SFC.

        :param thrust: Thrust (in N)
        :param engine_setting: Engine settings (climb, cruise,... )
        :param atmosphere: Atmosphere instance at intended altitude
        :return: SFC (in g/kw) and Power (in W)
        """
        # Load engine map and save interpolation formula
        rpm_vect = self.rpm_vect_ref
        pme_vect = self.pme_vect_ref
        sfc_matrix = self.sfc_matrix_ref
        torque_vect = pme_vect * 1e5 * self.volume / (8.0 * np.pi)
        ICE_sfc = interp2d(torque_vect, rpm_vect, sfc_matrix, kind="cubic")

        # Define RPM & mixture using engine settings
        if np.size(engine_setting) == 1:
            rpm_values = self.rpm_values[int(engine_setting)]
            mixture_values = self.mixture_values[int(engine_setting)]
        else:
            rpm_values = np.array(
                [self.rpm_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )
            mixture_values = np.array(
                [self.mixture_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )

        # Compute sfc @ 2500RPM
        real_power = np.zeros(np.size(thrust))
        torque = np.zeros(np.size(thrust))
        sfc = np.zeros(np.size(thrust))
        if np.size(thrust) == 1:
            real_power = (
                thrust * atmosphere.true_airspeed / self.propeller_efficiency(thrust, atmosphere)
            )
            torque = real_power / (rpm_values * np.pi / 30.0)
            sfc = ICE_sfc(torque, rpm_values) * mixture_values * self.k_factor_sfc
        else:
            for idx in range(np.size(thrust)):
                local_atmosphere = Atmosphere(
                    atmosphere.get_altitude()[idx], altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx]
                real_power[idx] = (
                    thrust[idx]
                    * atmosphere.true_airspeed[idx]
                    / self.propeller_efficiency(thrust[idx], local_atmosphere)
                )
                torque[idx] = real_power[idx] / (rpm_values[idx] * np.pi / 30.0)
                sfc[idx] = (
                    ICE_sfc(torque[idx], rpm_values[idx]) * mixture_values[idx] * self.k_factor_sfc
                )
        return sfc, real_power

    def max_thrust(
        self,
        engine_setting: Union[float, Sequence[float]],
        atmosphere: Atmosphere,
    ) -> np.ndarray:
        """
        Computation of maximum thrust either due to propeller thrust limit or ICE max power.

        :param engine_setting: Engine settings (climb, cruise,... )
        :param atmosphere: Atmosphere instance at intended altitude (should be <=20km)
        :return: maximum thrust (in N)
        """
        # Calculate maximum propeller thrust @ given altitude and speed
        if isinstance(atmosphere.true_airspeed, float):
            lower_bound = np.interp(atmosphere.true_airspeed, self.speed_SL, self.thrust_limit_SL)
            upper_bound = np.interp(atmosphere.true_airspeed, self.speed_CL, self.thrust_limit_CL)
        else:
            lower_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_SL, self.thrust_limit_SL
            )
            upper_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_CL, self.thrust_limit_CL
            )
        altitude = atmosphere.get_altitude(altitude_in_feet=False)
        thrust_max_propeller = (
            lower_bound
            + (upper_bound - lower_bound)
            * np.minimum(altitude, self.cruise_altitude_propeller)
            / self.cruise_altitude_propeller
        )

        # Calculate engine max power @ given RPM & altitude
        rpm_vect = self.rpm_vect_ref
        pme_limit_vect = self.pme_limit_vect_ref
        torque_vect = pme_limit_vect * 1e5 * self.volume / (8.0 * np.pi)
        power_max_vect = torque_vect * rpm_vect * (np.pi / 30.0)
        if np.size(engine_setting) == 1:
            rpm_values = np.array(self.rpm_values[int(engine_setting)])
            max_power_SL = np.interp(rpm_values, rpm_vect, power_max_vect)
        else:
            rpm_values = np.array(
                [self.rpm_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )
            max_power_SL = np.interp(list(rpm_values), rpm_vect, power_max_vect)
        sigma = atmosphere.density / Atmosphere(0.0).density
        max_power = max_power_SL * (sigma - (1 - sigma) / 7.55)

        # Found thrust relative to ICE maximum power @ given altitude and speed: calculates first
        # thrust interpolation vector (between min and max of propeller table) and associated
        # efficiency, then calculates power and found thrust (interpolation limits to max
        # propeller thrust)
        thrust_interp = np.linspace(
            np.min(self.thrust_SL) * np.ones(np.size(thrust_max_propeller)),
            thrust_max_propeller,
            10,
        ).transpose()
        if np.size(altitude) == 1:  # Calculate for float
            thrust_max_global = 0.0
            local_atmosphere = Atmosphere(
                altitude * np.ones(np.size(thrust_interp)), altitude_in_feet=False
            )
            local_atmosphere.mach = atmosphere.mach * np.ones(np.size(thrust_interp))
            propeller_efficiency = self.propeller_efficiency(thrust_interp[0], local_atmosphere)
            mechanical_power = thrust_interp[0] * atmosphere.true_airspeed / propeller_efficiency
            if np.min(mechanical_power) > max_power:
                efficiency_relative_error = 1
                propeller_efficiency = propeller_efficiency[0]
                while efficiency_relative_error > 1e-2:
                    thrust_max_global = max_power * propeller_efficiency / atmosphere.true_airspeed
                    propeller_efficiency_new = self.propeller_efficiency(
                        thrust_max_global, atmosphere
                    )
                    efficiency_relative_error = np.abs(
                        (propeller_efficiency_new - propeller_efficiency)
                        / efficiency_relative_error
                    )
                    propeller_efficiency = propeller_efficiency_new
            else:
                thrust_max_global = np.interp(max_power, mechanical_power, thrust_interp[0])
        else:  # Calculate for array
            thrust_max_global = np.zeros(np.size(altitude))
            for idx in range(np.size(altitude)):
                local_atmosphere = Atmosphere(
                    altitude[idx] * np.ones(np.size(thrust_interp[idx])), altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx] * np.ones(np.size(thrust_interp[idx]))
                propeller_efficiency = self.propeller_efficiency(
                    thrust_interp[idx], local_atmosphere
                )
                mechanical_power = (
                    thrust_interp[idx] * atmosphere.true_airspeed[idx] / propeller_efficiency
                )
                if (
                    np.min(mechanical_power) > max_power[idx]
                ):  # take the lower bound efficiency for calculation
                    efficiency_relative_error = 1
                    local_atmosphere = Atmosphere(altitude[idx], altitude_in_feet=False)
                    local_atmosphere.mach = atmosphere.mach[idx]
                    propeller_efficiency = propeller_efficiency[0]
                    while efficiency_relative_error > 1e-2:
                        thrust_max_global[idx] = (
                            max_power[idx] * propeller_efficiency / atmosphere.true_airspeed[idx]
                        )
                        propeller_efficiency_new = self.propeller_efficiency(
                            thrust_max_global[idx], local_atmosphere
                        )
                        efficiency_relative_error = np.abs(
                            (propeller_efficiency_new - propeller_efficiency)
                            / efficiency_relative_error
                        )
                        propeller_efficiency = propeller_efficiency_new
                else:
                    thrust_max_global[idx] = np.interp(
                        max_power[idx], mechanical_power, thrust_interp[idx]
                    )

        return thrust_max_global

    def compute_weight(self) -> float:
        """
        Computes weight of installed propulsion (engine, nacelle and propeller) depending on
        maximum power. Uses model described in : Gudmundsson, Snorri. General aviation aircraft
        design: Applied Methods and Procedures. Butterworth-Heinemann, 2013. Equation (6-44)

        """
        power_sl = self.max_power / 745.7  # conversion to european hp
        uninstalled_weight = (power_sl - 21.55) / 0.5515
        self.engine.mass = uninstalled_weight

        return uninstalled_weight

    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle) from maximum power.
        Model from :...

        """

        # Compute engine dimensions
        self.engine.length = self.ref["length"] * (self.max_power / self.ref["max_power"]) ** (
            1 / 3
        )
        self.engine.height = self.ref["height"] * (self.max_power / self.ref["max_power"]) ** (
            1 / 3
        )
        self.engine.width = self.ref["width"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)

        if self.prop_layout == 3.0:
            nacelle_length = 1.15 * self.engine.length
            # Based on the length between nose and firewall for TB20 and SR22
        else:
            nacelle_length = 2.0 * self.engine.length

        # Compute nacelle dimensions
        self.nacelle = Nacelle(
            height=self.engine.height * 1.1,
            width=self.engine.width * 1.1,
            length=nacelle_length,
        )
        self.nacelle.wet_area = 2 * (self.nacelle.height + self.nacelle.width) * self.nacelle.length

        return (
            self.nacelle["height"],
            self.nacelle["width"],
            self.nacelle["length"],
            self.nacelle["wet_area"],
        )

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        """
        Compute nacelle drag coefficient cd0.

        """

        # Compute dimensions
        _, _, _, _ = self.compute_dimensions()
        # Local Reynolds:
        reynolds = unit_reynolds * self.nacelle.length
        # Roskam method for wing-nacelle interaction factor (vol 6 page 3.62)
        cf_nac = 0.455 / (
            (1 + 0.144 * mach ** 2) ** 0.65 * (np.log10(reynolds)) ** 2.58
        )  # 100% turbulent
        fineness_ratio = self.nacelle.length / np.sqrt(
            4 * self.nacelle.height * self.nacelle.width / np.pi
        )
        ff_nac = 1 + 0.35 / fineness_ratio  # Raymer (seen in Gudmunsson)
        if_nac = 1.2  # Jenkinson (seen in Gudmundsson)
        drag_force = cf_nac * ff_nac * self.nacelle.wet_area * if_nac

        # Roskam part 6 chapter 4.5.2.1 with no incidence
        interference_drag = 0.036 * wing_mac * self.nacelle.width * 0.2 ** 2.0

        # The interference drag is for the nacelle/wing interference, since for fuselage mounted
        # engine the nacelle drag is not taken into account we can do like so
        return drag_force + interference_drag


@AddKeyAttributes(ENGINE_LABELS)
class Engine(DynamicAttributeDict):
    """
    Class for storing data for engine.

    An instance is a simple dict, but for convenience, each item can be accessed
    as an attribute (inspired by pandas DataFrames). Hence, one can write::

        >>> engine = Engine(power_SL=10000.)
        >>> engine["power_SL"]
        10000.0
        >>> engine["mass"] = 70000.
        >>> engine.mass
        70000.0
        >>> engine.mass = 50000.
        >>> engine["mass"]
        50000.0

    Note: constructor will forbid usage of unknown keys as keyword argument, but
    other methods will allow them, while not making the matching between dict
    keys and attributes, hence::

        >>> engine["foo"] = 42  # Ok
        >>> bar = engine.foo  # raises exception !!!!
        >>> engine.foo = 50  # allowed by Python
        >>> # But inner dict is not affected:
        >>> engine.foo
        50
        >>> engine["foo"]
        42

    This class is especially useful for generating pandas DataFrame: a pandas
    DataFrame can be generated from a list of dict... or a list of oad.FlightPoint
    instances.

    The set of dictionary keys that are mapped to instance attributes is given by
    the :meth:`get_attribute_keys`.
    """


@AddKeyAttributes(NACELLE_LABELS)
class Nacelle(DynamicAttributeDict):
    """
    Class for storing data for nacelle.

    Similar to :class:`Engine`.
    """
