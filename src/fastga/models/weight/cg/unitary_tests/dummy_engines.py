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

from typing import Union

import pandas as pd
from openmdao.core.component import Component
import numpy as np

import fastoad.api as oad

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion
from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet

ENGINE_WRAPPER_BE76 = "test.wrapper.cg.beechcraft.dummy_engine"
ENGINE_WRAPPER_SR22 = "test.wrapper.cg.cirrus.dummy_engine"
ENGINE_WRAPPER_TBM900 = "test.wrapper.cg.daher.dummy_engine"


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

    def compute_flight_points(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]):
        flight_points.thrust = 3500.0
        flight_points.sfc = 0.0

    def compute_weight(self) -> float:
        return 562.83 / 2.0

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(
        self,
        mach: Union[float, np.array],
        unit_reynolds: Union[float, np.array],
        wing_mac: float,
    ) -> Union[float, np.array]:
        return 0.0

    def get_consumed_mass(self, flight_point: oad.FlightPoint, time_step: float) -> float:
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

    def compute_flight_points(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]):
        flight_points.thrust = 5417.0
        flight_points.sfc = 0.0

    def compute_weight(self) -> float:
        return 331.88

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(
        self,
        mach: Union[float, np.array],
        unit_reynolds: Union[float, np.array],
        wing_mac: float,
    ) -> Union[float, np.array]:
        return 0.0

    def get_consumed_mass(self, flight_point: oad.FlightPoint, time_step: float) -> float:
        return 0.0

    def compute_max_power(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]) -> float:
        return 0.0


@oad.RegisterPropulsion(ENGINE_WRAPPER_SR22)
class DummyEngineWrapperSR22(oad.IOMPropulsionWrapper):
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
            DummyEngineSR22(**engine_params), inputs["data:geometry:propulsion:engine:count"]
        )


####################################################################################################
# Daher TBM900 dummy engine ########################################################################
####################################################################################################


class DummyEngineTBM900(AbstractFuelPropulsion):
    def __init__(
        self,
        fuel_type: float,
        prop_layout: float,
    ):
        """
        Dummy engine model returning nacelle aerodynamic drag force.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.fuel_type = fuel_type

    def compute_flight_points(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]):
        flight_points.thrust = 8000.0
        flight_points.sfc = 0.0

    def compute_weight(self) -> float:
        return 455.915416

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(
        self,
        mach: Union[float, np.array],
        unit_reynolds: Union[float, np.array],
        wing_mac: float,
    ) -> Union[float, np.array]:
        return 0.0

    def get_consumed_mass(self, flight_point: oad.FlightPoint, time_step: float) -> float:
        return 0.0

    def compute_max_power(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]) -> float:
        return 0.0


@oad.RegisterPropulsion(ENGINE_WRAPPER_TBM900)
class DummyEngineWrapperTBM900(oad.IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:propulsion:fuel_type", np.nan)
        component.add_input("data:geometry:propulsion:engine:layout", np.nan)
        component.add_input("data:geometry:propulsion:engine:count", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        engine_params = {
            "fuel_type": inputs["data:propulsion:fuel_type"],
            "prop_layout": inputs["data:geometry:propulsion:engine:layout"],
        }

        return FuelEngineSet(
            DummyEngineTBM900(**engine_params), inputs["data:geometry:propulsion:engine:count"]
        )
