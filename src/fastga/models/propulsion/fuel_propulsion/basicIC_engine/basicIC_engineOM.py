"""Parametric propeller IC engine."""
# -*- coding: utf-8 -*-
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

import os.path as pth
import logging
import math
import numpy as np
import pandas as pd
from typing import Union, Sequence, Tuple, Optional
from scipy.constants import g
from pyfmi import load_fmu

from .exceptions import FastBasicICEngineInconsistentInputParametersError
from models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes

from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError

from . import resources

# Logger for this module
_LOGGER = logging.getLogger(__name__)

PROPELLER_EFFICIENCY = 0.83  # Used to be 0.8 maybe make it an xml parameter
PROPULSION_FMU = "ICengineFMU.fmu"  # "ICengineFMU.fmu"

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
# Set of dictionary keys that are mapped to instance attributes.
PROPELLER_LABELS = {
    "area": dict(doc="Area in meters²."),
    "depth": dict(doc="Depth in meters."),
    "diameter": dict(doc="Diameter in meters."),
    "thrust_SL": dict(doc="Fixed point thrust at sea level in kilograms."),
}


class BasicICEngine(AbstractFuelPropulsion):

    def __init__(
            self,
            max_power: float,
            design_altitude: float,
            design_speed: float,
            fuel_type: float,
            strokes_nb: float,
            prop_layout: float,
    ):
        """
        Parametric Internal Combustion engine.

        It computes engine characteristics using fuel type, motor architecture
        and constant propeller efficiency using analytical model from following sources:

        :param max_power: maximum delivered mechanical power of engine (units=W)
        :param design_altitude: design altitude for cruise (units=m)
        :param design_speed: design altitude for cruise (units=m/s)
        :param fuel_type: 1.0 for gasoline and 2.0 for diesel engine and 3.0 for Jet Fuel
        :param strokes_nb: can be either 2-strockes (=2.0) or 4-strockes (=4.0)
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
        else:
            self.ref = {
                "max_power": 160000,
                "length": 0.859,
                "height": 0.659,
                "width": 0.650,
                "mass": 205,
            }  # TDA CR 1.9 16V
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude = design_altitude
        self.design_speed = design_speed
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb
        self.idle_thrust_rate = 0.01

        # Declare sub-components attribute
        self.engine = Engine(power_SL=max_power)
        self.nacelle = None
        self.propeller = None

        # This dictionary is expected to have a Mixture coefficient for all EngineSetting values
        self.mixture_values = {
            EngineSetting.TAKEOFF: 1.5,
            EngineSetting.CLIMB: 1.5,
            EngineSetting.CRUISE: 1.0,
            EngineSetting.IDLE: 1.0,
        }

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.mixture_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", unknown_keys)

        # Define the FMU model used
        # noinspection PyProtectedMember
        self.model = load_fmu(pth.join(resources.__path__._path[0], PROPULSION_FMU))

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        # pylint: disable=too-many-arguments  # they define the trajectory
        sfc, thrust_rate, thrust = self._compute_flight_points(
            flight_points.mach,
            flight_points.altitude,
            flight_points.engine_setting,
            flight_points.thrust_is_regulated,
            flight_points.thrust_rate,
            flight_points.thrust,
        )
        flight_points['sfc'] = sfc
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
        Computes the Specific Fuel Consumption based on aircraft trajectory conditions.
        
        :param flight_points.mach: Mach number
        :param flight_points.altitude: (unit=m) altitude w.r.t. to sea level
        :param flight_points.engine_setting: define
        :param flight_points.thrust_is_regulated: tells if thrust_rate or thrust should be used (works element-wise)
        :param flight_points.thrust_rate: thrust rate (unit=none)
        :param flight_points.thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)
        """

        # Treat inputs (with check on thrust rate <=1.0)
        mach = np.asarray(mach)
        altitude = np.asarray(altitude)
        if thrust_is_regulated is not None:
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_is_regulated, thrust_rate, thrust = self._check_thrust_inputs(
            thrust_is_regulated, thrust_rate, thrust
        )
        thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_rate = np.asarray(thrust_rate)
        thrust = np.asarray(thrust)

        # Get maximum thrust @ given altitude
        atmosphere = Atmosphere(altitude, altitude_in_feet=False)
        max_thrust = self.max_thrust(atmosphere, mach)

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
            out_thrust[thrust_is_regulated] = np.minimum(out_thrust[thrust_is_regulated],
                                                         max_thrust[thrust_is_regulated])

        # thrust_rate is obtained from entire thrust vector (could be optimized if needed,
        # as some thrust rates that are computed may have been provided as input)
        out_thrust_rate = out_thrust / max_thrust

        # Resetting the model is required to re-run from scratch the FMU
        self.model.reset()

        # Evaluate max power
        altitude = atmosphere.get_altitude(altitude_in_feet=True)
        sigma = Atmosphere(altitude).density / Atmosphere(0.0).density
        max_power = (self.max_power / 1e3) * (sigma - (1 - sigma) / 7.55)  # max power in kW

        # Set all parameters before computation
        self.model.set("maximum_power", max_power)
        self.model.set("fuel_type", self.fuel_type)
        self.model.set("strokes_nb", self.strokes_nb)
        self.model.set("propeller_efficiency", PROPELLER_EFFICIENCY)
        self.model.set("thrust.k", out_thrust)
        self.model.set("mach.k", mach)
        self.model.initialize()

        result = self.model.simulate(
            start_time=0.0,
            final_time=0.0,
            options={"ncp": 2},
        )

        sfc = result['sfc.y'][-1] / np.maximum(out_thrust, 1e-6)  # avoid 0 division

        return sfc, out_thrust_rate, out_thrust

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

    def max_thrust(
            self,
            atmosphere: Atmosphere,
            mach: Union[float, Sequence[float]],
    ) -> np.ndarray:
        """
        Computation of maximum thrust.
        Uses model described in ...
        :param atmosphere: Atmosphere instance at intended altitude (should be <=20km)
        :param mach: Mach number(s) (should be between 0.05 and 1.0)
        :return: maximum thrust (in N)
        """

        # Calculate maximum mechanical power @ given altitude
        altitude = atmosphere.get_altitude(altitude_in_feet=True)
        mach = np.asarray(mach)
        sigma = Atmosphere(altitude).density / Atmosphere(0.0).density
        max_power = self.max_power * (sigma - (1 - sigma) / 7.55)
        _, _, _, _, _, _ = self.compute_dimensions()
        thrust_1 = (self.propeller["thrust_SL"] * g) * sigma ** (1 / 3)  # considered fixed point @altitude
        thrust_2 = max_power * PROPELLER_EFFICIENCY / np.maximum(mach * Atmosphere(altitude).speed_of_sound, 1e-20)

        return np.minimum(thrust_1, thrust_2)

    def compute_weight(self) -> float:
        """
        Computes weight of installed propulsion (engine, nacelle and propeller) depending on maximum power.
        Uses model described in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
        Butterworth-Heinemann, 2013. Equation (6-44)

        """

        power_sl = self.max_power / 745.7  # conversion to european hp
        uninstalled_weight = ((power_sl - 21.55) / 0.5515)
        self.engine.mass = uninstalled_weight

        return uninstalled_weight

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle/propeller) from maximum power.
        Model from :...

        """

        # Compute engine dimensions
        self.engine.length = self.ref["length"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)
        self.engine.height = self.ref["height"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)
        self.engine.width = self.ref["width"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)

        if self.prop_layout == 3.0:
            nacelle_length = 1.15 * self.engine.length
            # Based on the length between nose and firewall for TB20 and SR22
        else:
            nacelle_length = 1.50 * self.engine.length

        # Compute nacelle dimensions
        self.nacelle = Nacelle(
            height=self.engine.height * 1.1,
            width=self.engine.width * 1.1,
            length=nacelle_length,
        )
        self.nacelle.wet_area = 2 * (self.nacelle.height + self.nacelle.width) * self.nacelle.length

        # Compute propeller dimensions (2-blades)
        w_propeller = 2500  # regulated propeller speed in RPM
        v_sound = Atmosphere(self.design_altitude, altitude_in_feet=False).speed_of_sound
        d_max = (((v_sound * 0.85) ** 2 - self.design_speed ** 2) / ((w_propeller * math.pi / 30) / 2) ** 2) ** 0.5
        d_opt = 1.04 ** 2 * ((self.max_power / 735.5) * 1e8 / (w_propeller ** 2 * self.design_speed * 3.6)) ** (1 / 4)
        d = min(d_max, d_opt)
        t_0 = 7.4 * ((self.max_power / 735.5) * d) ** (2 / 3)
        area = 13307 * t_0 / (d ** 2 * w_propeller ** 2)
        chord = area / d
        self.propeller = Propeller(
            area=area,
            depth=chord * 1.1,
            diameter=d,
            thrust_SL=t_0,
        )
        propeller_depth = max(chord*1.1, 0.2*d)
        # For clarity purposes, it has been assimilated as the spinner length

        return self.nacelle["height"], self.nacelle["width"], self.nacelle["length"], self.nacelle[
            "wet_area"], self.propeller["diameter"], self.propeller["depth"]

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        """
        Compute nacelle drag coefficient cd0.

        """

        # Compute dimensions
        _, _, _, _, _, _ = self.compute_dimensions()
        # Local Reynolds:
        reynolds = unit_reynolds * self.nacelle.length
        # Roskam method for wing-nacelle interaction factor (vol 6 page 3.62)
        cf_nac = 0.455 / ((1 + 0.144 * mach ** 2) ** 0.65 * (math.log10(reynolds)) ** 2.58)  # 100% turbulent
        f = self.nacelle.length / math.sqrt(4 * self.nacelle.height * self.nacelle.width / math.pi)
        ff_nac = 1 + 0.35 / f  # Raymer (seen in Gudmunsson)
        if_nac = 0.036 * self.nacelle.width * wing_mac * 0.04
        drag_force = (cf_nac * ff_nac * self.nacelle.wet_area + if_nac)

        return drag_force


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
    DataFrame can be generated from a list of dict... or a list of FlightPoint
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


@AddKeyAttributes(PROPELLER_LABELS)
class Propeller(DynamicAttributeDict):
    """
    Class for storing data for propeller.

    Similar to :class:`Engine`.
    """
