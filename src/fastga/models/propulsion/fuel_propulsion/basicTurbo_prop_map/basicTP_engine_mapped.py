"""Parametric turboprop engine."""
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
from typing import Union, Sequence, Tuple, Optional
import numpy as np
from scipy.interpolate import interp2d, interp1d

import fastoad.api as oad
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError

from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop_map.exceptions import (
    FastBasicICEngineInconsistentInputParametersError,
)
from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop.basicTP_engine import BasicTPEngine
from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes

# Logger for this module
_LOGGER = logging.getLogger(__name__)

# Set of dictionary keys that are mapped to instance attributes.
ENGINE_LABELS = {
    "power_SL": dict(doc="power at sea level in watts."),
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


class BasicTPEngineMapped(AbstractFuelPropulsion):

    """Mapped version of the turboprop, need the constructed table beforehand to work."""

    def __init__(
        self,
        power_design: float,  # In kW
        t_41t_design: float,
        opr_design: float,
        cruise_altitude_propeller: float,
        design_altitude: float,
        design_mach: float,
        prop_layout: float,
        bleed_control: float,
        itt_limit: float,
        power_limit: float,
        opr_limit: float,
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
        turbo_mach_SL,
        turbo_thrust_SL,
        turbo_thrust_max_SL,
        turbo_sfc_SL,
        turbo_mach_CL,
        turbo_thrust_CL,
        turbo_thrust_max_CL,
        turbo_sfc_CL,
        turbo_mach_IL,
        turbo_thrust_IL,
        turbo_thrust_max_IL,
        turbo_sfc_IL,
        level_IL,
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    ):
        """
        Parametric turboprop engine reading a map computed beforehand. Based on the basic turboprop
        engine, but post treatment is done beforehand to save computation time on the sfc estimation
        function

        :param power_design: thermodynamic power at the design point, in kW
        :param t_41t_design: turbine entry temperature at the design point, in K
        :param opr_design: overall pressure ratio at the design point
        :param cruise_altitude_propeller: cruise altitude, in m
        :param design_altitude: altitude at the design point, in m
        :param design_mach: mach number at the design point
        :param prop_layout: location of the turboprop
        :param bleed_control: usage of the bleed in off-design point, "low" or "high"
        :param itt_limit: temperature limit between the turbines, in K
        :param power_limit: power limit on the gearbox, in kW
        :param opr_limit: opr limit in the compressor
        :param speed_SL: array with the speed at which the sea level performance of the propeller
        were computed
        :param thrust_SL: array with the required thrust at which the sea level performance of the
        propeller were
        computed
        :param thrust_limit_SL: array with the limit thrust available at the speed in speed_SL
        :param efficiency_SL: array containing the sea level efficiency computed at speed_SL and
        thrust_SL
        :param speed_CL: array with the speed at which the cruise level performance of the propeller
        were computed
        :param thrust_CL: array with the required thrust at which the cruise level performance of
        the propeller were  computed
        :param thrust_limit_CL: array with the limit thrust available at the speed in speed_CL
        :param efficiency_CL: array containing the cruise level efficiency computed at speed_CL and
        thrust_CL
        :param turbo_mach_SL: array with the mach at which the sea level performance of the
        turboprop were computed
        :param turbo_thrust_SL: array with the required thrust at which the sea level performance of
        the turboprop were computed
        :param turbo_thrust_max_SL: array with the limit thrust available at the mach in
        turbo_mach_SL
        :param turbo_sfc_SL: array containing the sea level sfc computed at turbo_mach_SL and
        turbo_thrust_SL
        :param turbo_mach_CL: array with the mach at which the cruise level performance of the
        turboprop were computed
        :param turbo_thrust_CL: array with the required thrust at which the cruise level performance
        of the turboprop were computed
        :param turbo_thrust_max_CL: array with the limit thrust available at the mach in
        turbo_mach_CL
        :param turbo_sfc_CL: array containing the cruise level sfc computed at turbo_mach_CL and
        turbo_thrust_CL
        :param turbo_mach_IL: array with the mach at which the intermediate level performance of the
        turboprop were computed
        :param turbo_thrust_IL: array with the required thrust at which the intermediate level
        performance of the turboprop were computed
        :param turbo_thrust_max_IL: array with the limit thrust available at the mach in
        turbo_mach_IL
        :param turbo_sfc_IL: array containing the intermediate level sfc computed at turbo_mach_IL
        and turbo_thrust_IL
        :param level_IL: altitude at which the intermediate level computation were conducted
        """

        self.turboprop = BasicTPEngine(
            power_design=power_design,
            t_41t_design=t_41t_design,
            opr_design=opr_design,
            cruise_altitude_propeller=cruise_altitude_propeller,
            design_altitude=design_altitude,
            design_mach=design_mach,
            prop_layout=prop_layout,
            bleed_control=bleed_control,
            itt_limit=itt_limit,
            power_limit=power_limit,
            opr_limit=opr_limit,
            speed_SL=speed_SL,
            thrust_SL=thrust_SL,
            thrust_limit_SL=thrust_limit_SL,
            efficiency_SL=efficiency_SL,
            speed_CL=speed_CL,
            thrust_CL=thrust_CL,
            thrust_limit_CL=thrust_limit_CL,
            efficiency_CL=efficiency_CL,
            effective_J=effective_J,
            effective_efficiency_ls=effective_efficiency_ls,
            effective_efficiency_cruise=effective_efficiency_cruise,
            eta_225=eta_225,
            eta_253=eta_253,
            eta_445=eta_445,
            eta_455=eta_455,
            eta_q=eta_q,
            eta_axe=eta_axe,
            pi_02=pi_02,
            pi_cc=pi_cc,
            cooling_ratio=cooling_ratio,
            hp_shaft_power_out=hp_shaft_power_out,
            gearbox_efficiency=gearbox_efficiency,
            inter_compressor_bleed=inter_compressor_bleed,
            exhaust_mach_design=exhaust_mach_design,
            pr_1_ratio_design=pr_1_ratio_design,
        )

        self.speed_SL = speed_SL
        self.thrust_SL = thrust_SL
        self.thrust_limit_SL = thrust_limit_SL
        self.efficiency_SL = efficiency_SL
        self.speed_CL = speed_CL
        self.thrust_CL = thrust_CL
        self.thrust_limit_CL = thrust_limit_CL
        self.efficiency_CL = efficiency_CL

        formatted_turbo_thrust_SL, formatted_sfc_SL = reformat_table(turbo_thrust_SL, turbo_sfc_SL)
        self.turbo_mach_SL = turbo_mach_SL
        self.turbo_thrust_SL = formatted_turbo_thrust_SL
        self.turbo_thrust_max_SL = turbo_thrust_max_SL
        self.turbo_sfc_SL = formatted_sfc_SL

        formatted_turbo_thrust_CL, formatted_sfc_CL = reformat_table(turbo_thrust_CL, turbo_sfc_CL)
        self.turbo_mach_CL = turbo_mach_CL
        self.turbo_thrust_CL = formatted_turbo_thrust_CL
        self.turbo_thrust_max_CL = turbo_thrust_max_CL
        self.turbo_sfc_CL = formatted_sfc_CL
        self.cruise_altitude_propeller = float(cruise_altitude_propeller)

        formatted_turbo_thrust_IL, formatted_sfc_IL = reformat_table(turbo_thrust_IL, turbo_sfc_IL)
        self.turbo_mach_IL = turbo_mach_IL
        self.turbo_thrust_IL = formatted_turbo_thrust_IL
        self.turbo_thrust_max_IL = turbo_thrust_max_IL
        self.turbo_sfc_IL = formatted_sfc_IL
        self.intermediate_altitude = float(level_IL)

        self.specific_shape = None

        # Declare sub-components attribute
        self.engine = Engine(power_SL=power_design)
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

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.mixture_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", unknown_keys)

    def compute_flight_points(self, flight_points: oad.FlightPoint):
        # pylint: disable=too-many-arguments
        # they define the trajectory
        self.specific_shape = np.shape(flight_points.mach)
        if isinstance(flight_points.mach, float):
            sfc, thrust_rate, thrust = self._compute_flight_points(
                flight_points.mach,
                flight_points.altitude,
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
        thrust_is_regulated: Optional[Union[bool, Sequence]] = None,
        thrust_rate: Optional[Union[float, Sequence]] = None,
        thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence]]:
        """
        Same as :meth:`compute_flight_points`.

        :param mach: Mach number
        :param altitude: (unit=m) altitude w.r.t. to sea level
        :param thrust_is_regulated: tells if thrust_rate or thrust should be used
        (works element-wise)
        :param thrust_rate: thrust rate (unit=none)
        :param thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)

        Computes the Specific Fuel Consumption based on aircraft trajectory conditions.
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
        max_thrust = self.max_thrust(atmosphere)

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

        # Now SFC (g/s/N) can be computed and converted to sfc_thrust (kg/N) to match computation
        # from turboshaft
        sfc_thrust = self.sfc(out_thrust, atmosphere)

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

    def compute_max_power(self, flight_points: oad.FlightPoint) -> Union[float, Sequence]:
        """
        Compute the turboprop maximum power @ given flight-point. Uses the original method

        :param flight_points: current flight point, with altitude in meters as always !
        :return: maximum power in kW
        """

        return self.turboprop.compute_max_power(flight_points)

    def read_sfc_table(self, thrust: float, atmosphere: float) -> float:
        """Reads the turboprop table and gives corresponding sfc."""
        altitude = atmosphere.get_altitude(altitude_in_feet=False)
        if altitude > self.intermediate_altitude:
            mach_il = np.clip(atmosphere.mach, 1e-5, max(self.turbo_mach_IL))
            mach_cl = np.clip(atmosphere.mach, 1e-5, max(self.turbo_mach_CL))
            sfc_interp_IL = interp2d(self.turbo_thrust_IL, self.turbo_mach_IL, self.turbo_sfc_IL)
            max_thrust_interp_IL = interp1d(self.turbo_mach_IL, self.turbo_thrust_max_IL)
            sfc_interp_CL = interp2d(self.turbo_thrust_CL, self.turbo_mach_CL, self.turbo_sfc_CL)
            max_thrust_interp_CL = interp1d(self.turbo_mach_CL, self.turbo_thrust_max_CL)
            thrust_interp_IL = np.clip(
                thrust, min(self.turbo_thrust_IL), max_thrust_interp_IL(mach_il)
            )
            thrust_interp_CL = np.clip(
                thrust, min(self.turbo_thrust_CL), max_thrust_interp_CL(mach_cl)
            )
            lower_bound = float(sfc_interp_IL(thrust_interp_IL, mach_il))
            upper_bound = float(sfc_interp_CL(thrust_interp_CL, mach_cl))
            sfc = float(
                interp1d(
                    [self.intermediate_altitude, self.cruise_altitude_propeller],
                    [lower_bound, upper_bound],
                )(max(float(altitude), 0.0))
            )
        else:
            mach_sl = np.clip(atmosphere.mach, 1e-5, max(self.turbo_mach_SL))
            mach_il = np.clip(atmosphere.mach, 1e-5, max(self.turbo_mach_IL))
            sfc_interp_SL = interp2d(self.turbo_thrust_SL, self.turbo_mach_SL, self.turbo_sfc_SL)
            max_thrust_interp_SL = interp1d(self.turbo_mach_SL, self.turbo_thrust_max_SL)
            sfc_interp_IL = interp2d(self.turbo_thrust_IL, self.turbo_mach_IL, self.turbo_sfc_IL)
            max_thrust_interp_IL = interp1d(self.turbo_mach_IL, self.turbo_thrust_max_IL)
            thrust_interp_SL = np.clip(
                thrust, min(self.turbo_thrust_SL), max_thrust_interp_SL(mach_sl)
            )
            thrust_interp_IL = np.clip(
                thrust, min(self.turbo_thrust_IL), max_thrust_interp_IL(mach_il)
            )
            lower_bound = float(sfc_interp_SL(thrust_interp_SL, mach_sl))
            upper_bound = float(sfc_interp_IL(thrust_interp_IL, mach_il))
            sfc = float(
                interp1d([0.0, self.intermediate_altitude], [lower_bound, upper_bound])(
                    max(float(altitude), 0.0)
                )
            )

        return sfc

    def sfc(
        self,
        thrust: Union[float, Sequence[float]],
        atmosphere: Atmosphere,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computation of the SFC.

        :param thrust: Thrust (in N)
        :param atmosphere: Atmosphere instance at intended altitude
        :return: SFC (in kg/s/N)
        """

        # torque = np.zeros(np.size(thrust))
        sfc = np.zeros(np.size(thrust))
        if np.size(thrust) == 1:
            sfc = self.read_sfc_table(thrust, atmosphere)
        else:
            for idx in range(np.size(thrust)):
                local_atmosphere = Atmosphere(
                    atmosphere.get_altitude(altitude_in_feet=False)[idx], altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx]
                sfc[idx] = self.read_sfc_table(thrust[idx], local_atmosphere)

        return sfc

    def max_thrust(
        self,
        atmosphere: Atmosphere,
    ) -> np.ndarray:
        """
        Computation of maximum thrust either due to propeller thrust limit or turboprop max power.

        :param atmosphere: Atmosphere instance at intended altitude (should be <=20km)
        :return: maximum thrust (in N)
        """

        altitude = atmosphere.get_altitude(altitude_in_feet=False)
        if np.size(altitude) == 1:
            mach = atmosphere.mach
            max_thrust = self._max_thrust(altitude, mach)
            if isinstance(max_thrust, float):
                max_thrust = np.array([max_thrust])
        else:
            max_thrust = np.zeros(np.size(altitude))
            mach_array = atmosphere.mach
            for idx in range(np.size(altitude)):
                max_thrust[idx] = self._max_thrust(altitude[idx], mach_array[idx])

        return max_thrust

    def _max_thrust(self, altitude: float, mach: float) -> np.ndarray:

        if altitude > self.intermediate_altitude:
            max_thrust_interp_IL = interp1d(self.turbo_mach_IL, self.turbo_thrust_max_IL)
            max_thrust_interp_CL = interp1d(self.turbo_mach_CL, self.turbo_thrust_max_CL)
            max_thrust_IL = float(
                max_thrust_interp_IL(np.clip(mach, 1e-5, max(self.turbo_mach_IL)))
            )
            max_thrust_CL = float(
                max_thrust_interp_CL(np.clip(mach, 1e-5, max(self.turbo_mach_CL)))
            )
            max_thrust = float(
                interp1d(
                    [self.intermediate_altitude, self.cruise_altitude_propeller],
                    [max_thrust_IL, max_thrust_CL],
                )(max(float(altitude), 0.0))
            )
        else:
            max_thrust_interp_SL = interp1d(self.turbo_mach_SL, self.turbo_thrust_max_SL)
            max_thrust_interp_IL = interp1d(self.turbo_mach_CL, self.turbo_thrust_max_CL)
            max_thrust_SL = float(
                max_thrust_interp_SL(np.clip(mach, 1e-5, max(self.turbo_mach_SL)))
            )
            max_thrust_IL = float(
                max_thrust_interp_IL(np.clip(mach, 1e-5, max(self.turbo_mach_IL)))
            )
            max_thrust = float(
                interp1d([0.0, self.intermediate_altitude], [max_thrust_SL, max_thrust_IL])(
                    max(float(altitude), 0.0)
                )
            )

        return max_thrust

    def propeller_efficiency(
        self, thrust: Union[float, Sequence[float]], atmosphere: Atmosphere
    ) -> Union[float, Sequence]:
        """
        Compute the propeller efficiency. Should only take the thrust of one propeller.

        :param thrust: Thrust (in N)
        :param atmosphere: Atmosphere instance at intended altitude
        :return: efficiency
        """

        return self.turboprop.propeller_efficiency(thrust, atmosphere)

    def compute_weight(self) -> float:
        """
        Computes weight of uninstalled propulsion depending on maximum power. Uses model
        described in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and
        Procedures. Butterworth-Heinemann, 2013. Equation (6-44).
        """

        return self.turboprop.compute_weight()

    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle) from maximum power. Model from a
        regression on the PT6 family.
        """

        return self.turboprop.compute_dimensions()

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        """
        Compute nacelle drag coefficient cd0.

        """

        return self.turboprop.compute_drag(mach, unit_reynolds, wing_mac)


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


def reformat_table(thrust_table, sfc_table):
    """Reformat to fit the OpenMDAO formalism."""
    valid_idx_array = np.where(thrust_table != 0.0)[0]
    last_valid_idx = max(valid_idx_array)
    thrust_table = thrust_table[:last_valid_idx]
    sfc_table = sfc_table[:, :last_valid_idx]

    return thrust_table, sfc_table
