"""
Test module for geometry functions of cg components.
"""
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

import pandas as pd
from openmdao.core.component import Component
from typing import Union
import numpy as np

from stdatm import Atmosphere

import fastoad.api as oad

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion
from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet

ENGINE_WRAPPER_BE76 = "test.wrapper.performances.beechcraft.dummy_engine"
ENGINE_WRAPPER_SR22 = "test.wrapper.performances.cirrus.dummy_engine"
ENGINE_WRAPPER_TBM900 = "test.wrapper.performances.daher.dummy_engine"


####################################################################################################
# Beechcraft BE76 dummy engine #####################################################################
####################################################################################################


class DummyEngineBE76(AbstractFuelPropulsion):
    def __init__(
        self,
        max_power: float,
        design_altitude_propeller: float,
        fuel_type: float,
        strokes_nb: float,
        prop_layout: float,
    ):
        """
        Dummy engine model returning nacelle aerodynamic drag force.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude_propeller = design_altitude_propeller
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb
        self.max_thrust = 3500.0

    def compute_flight_points(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]):

        mach = np.array(flight_points.mach)
        thrust = np.array(flight_points.thrust)
        sigma = Atmosphere(flight_points.altitude).density / Atmosphere(0.0).density
        max_power = self.max_power * (sigma - (1 - sigma) / 7.55)
        max_thrust = np.minimum(
            self.max_thrust * sigma ** (1.0 / 3.0),
            max_power
            * 0.8
            / np.maximum(mach * Atmosphere(flight_points.altitude).speed_of_sound, 1e-20),
        )
        if flight_points.thrust_rate is None:
            flight_points.thrust = min(max_thrust, thrust)
            flight_points.thrust_rate = thrust / max_thrust
        elif flight_points.thrust is None:
            flight_points.thrust = max_thrust * np.array(flight_points.thrust_rate)
        else:
            flight_points.thrust_rate = np.where(
                flight_points.thrust_is_regulated,
                np.maximum(thrust, np.full_like(thrust, max_thrust))
                / np.full_like(thrust, max_thrust),
                flight_points.thrust_rate,
            )
            flight_points.thrust = np.where(
                flight_points.thrust_is_regulated,
                flight_points.thrust,
                max_thrust * np.array(flight_points.thrust_rate),
            )

        sfc_pmax = 7.96359441e-08  # fixed whatever the thrust ratio, sfc for ONE 130kW engine !
        sfc = (
            sfc_pmax
            * flight_points.thrust_rate
            * mach
            * Atmosphere(flight_points.altitude).speed_of_sound
        )
        flight_points.sfc = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        return 0.0

    def compute_max_power(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]) -> float:
        return 0.0


@oad.RegisterPropulsion(ENGINE_WRAPPER_BE76)
class DummyEngineWrapperBE76(oad.IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:aerodynamics:propeller:cruise_level:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:engine:layout", np.nan)
        component.add_input("data:geometry:propulsion:engine:count", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "design_altitude_propeller": inputs[
                "data:aerodynamics:propeller:cruise_level:altitude"
            ],
            "fuel_type": inputs["data:propulsion:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:engine:layout"],
        }

        return FuelEngineSet(
            DummyEngineBE76(**engine_params), inputs["data:geometry:propulsion:engine:count"]
        )


####################################################################################################
# Cirrus SR22 dummy engine #########################################################################
####################################################################################################


class DummyEngineSR22(AbstractFuelPropulsion):
    def __init__(self):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.max_power = 231000.0
        self.max_thrust = 5417.0

    def compute_flight_points(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]):

        mach = np.array(flight_points.mach)
        thrust = np.array(flight_points.thrust)
        sigma = Atmosphere(flight_points.altitude).density / Atmosphere(0.0).density
        max_power = self.max_power * (sigma - (1 - sigma) / 7.55)
        max_thrust = np.minimum(
            self.max_thrust * sigma ** (1.0 / 3.0),
            max_power
            * 0.8
            / np.maximum(mach * Atmosphere(flight_points.altitude).speed_of_sound, 1e-20),
        )
        if flight_points.thrust_rate is None:
            flight_points.thrust = min(max_thrust, thrust)
            flight_points.thrust_rate = thrust / max_thrust
        elif flight_points.thrust is None:
            flight_points.thrust = max_thrust * np.array(flight_points.thrust_rate)
        else:
            flight_points.thrust_rate = np.where(
                flight_points.thrust_is_regulated,
                np.maximum(thrust, np.full_like(thrust, max_thrust))
                / np.full_like(thrust, max_thrust),
                flight_points.thrust_rate,
            )
            flight_points.thrust = np.where(
                flight_points.thrust_is_regulated,
                flight_points.thrust,
                max_thrust * np.array(flight_points.thrust_rate),
            )
        sfc_pmax = 8.5080e-08  # fixed whatever the thrust ratio, sfc for ONE 130kW engine !
        sfc = (
            sfc_pmax
            * flight_points.thrust_rate
            * mach
            * Atmosphere(flight_points.altitude).speed_of_sound
        )
        flight_points.sfc = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        return 0.0

    def compute_max_power(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]) -> float:
        return 0.0


@oad.RegisterPropulsion(ENGINE_WRAPPER_SR22)
class DummyEngineWrapperSR22(oad.IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:aerodynamics:propeller:cruise_level:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:engine:count", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return FuelEngineSet(DummyEngineSR22(), inputs["data:geometry:propulsion:engine:count"])


####################################################################################################
# Daher TBM900 dummy engine ########################################################################
####################################################################################################


class DummyEngineTBM900(AbstractFuelPropulsion):
    def __init__(self):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.max_power = 634000.0
        self.max_thrust = 30000.0

    def compute_flight_points(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]):

        mach = np.array(flight_points.mach)
        thrust = np.array(flight_points.thrust)
        sigma = Atmosphere(flight_points.altitude).density / Atmosphere(0.0).density
        max_power = self.max_power * (sigma - (1 - sigma) / 7.55)
        max_thrust = np.minimum(
            self.max_thrust * sigma ** (1.0 / 3.0),
            max_power
            * 0.8
            / np.maximum(mach * Atmosphere(flight_points.altitude).speed_of_sound, 1e-20),
        )
        if flight_points.thrust_rate is None:
            flight_points.thrust = min(max_thrust, thrust)
            flight_points.thrust_rate = thrust / max_thrust
        elif flight_points.thrust is None:
            flight_points.thrust = max_thrust * np.array(flight_points.thrust_rate)
        else:
            flight_points.thrust_rate = np.where(
                flight_points.thrust_is_regulated,
                np.maximum(thrust, np.full_like(thrust, max_thrust))
                / np.full_like(thrust, max_thrust),
                flight_points.thrust_rate,
            )
            flight_points.thrust = np.where(
                flight_points.thrust_is_regulated,
                flight_points.thrust,
                max_thrust * np.array(flight_points.thrust_rate),
            )
        sfc_pmax = 1.5e-5  # fixed whatever the thrust rate
        sfc = sfc_pmax * flight_points.thrust_rate
        flight_points.sfc = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        return 0.0

    def compute_max_power(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]) -> float:
        return 0.0


@oad.RegisterPropulsion(ENGINE_WRAPPER_TBM900)
class DummyEngineWrapperTBM900(oad.IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:aerodynamics:propeller:cruise_level:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:engine:count", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return FuelEngineSet(DummyEngineTBM900(), inputs["data:geometry:propulsion:engine:count"])
