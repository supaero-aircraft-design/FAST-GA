"""
Test module for geometry functions of cg components
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

import pandas as pd
from openmdao.core.component import Component
from typing import Union
import numpy as np

from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint
from fastoad.model_base.propulsion import IOMPropulsionWrapper
from fastoad.model_base.atmosphere import Atmosphere

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

ENGINE_WRAPPER_SR22 = "test.wrapper.performances.cirrus.dummy_engine"


class DummyEngineSR22(AbstractFuelPropulsion):

    def __init__(self):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.max_power = 231000.0
        self.max_thrust = 5417.0

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):

        altitude = float(Atmosphere(np.array(flight_points.altitude)).get_altitude(altitude_in_feet=True))
        mach = np.array(flight_points.mach)
        thrust = np.array(flight_points.thrust)
        sigma = Atmosphere(altitude).density / Atmosphere(0.0).density
        max_power = self.max_power * (sigma - (1 - sigma) / 7.55)
        max_thrust = min(
            self.max_thrust * sigma ** (1. / 3.),
            max_power * 0.8 / np.maximum(mach * Atmosphere(altitude).speed_of_sound, 1e-20)
        )
        if flight_points.thrust_rate is None:
            flight_points.thrust = min(max_thrust, float(thrust))
            flight_points.thrust_rate = float(thrust) / max_thrust
        else:
            flight_points.thrust = max_thrust * np.array(flight_points.thrust_rate)
        sfc_pmax = 8.5080e-08  # fixed whatever the thrust ratio, sfc for ONE 130kW engine !
        sfc = sfc_pmax * flight_points.thrust_rate * mach * Atmosphere(altitude).speed_of_sound
        flight_points.sfc = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        return 0.0


@RegisterPropulsion(ENGINE_WRAPPER_SR22)
class DummyEngineWrapperSR22(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return DummyEngineSR22()
