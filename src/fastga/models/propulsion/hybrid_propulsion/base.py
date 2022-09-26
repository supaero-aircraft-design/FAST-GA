"""Base classes for hybrid-electric propulsion models."""
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

from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from fastoad.model_base import FlightPoint

from fastga.models.propulsion.propulsion import IPropulsionCS23


class AbstractHybridPropulsion(IPropulsionCS23, ABC):
    """
    Propulsion models that consume energy and fuel should inherit from this one.

    In inheritors, :meth:`compute_flight_points` is expected to define
    "sfc" and "thrust" in computed FlightPoint instances.
    """

    def get_consumed_mass(self, flight_point: FlightPoint, time_step: float) -> float:
        return time_step * flight_point.sfc * flight_point.thrust

    def get_consumed_energy(self, flight_point: FlightPoint, time_step: float) -> float:
        return time_step * flight_point.battery_power

    @abstractmethod
    def compute_dimensions(self):
        """
        Computes propulsion sub-components dimensions.

        """


class HybridEngineSet(AbstractHybridPropulsion):
    def __init__(self, engine: IPropulsionCS23, engine_count):
        """
        Class for modelling an assembly of identical electric engines.

        Thrust is supposed equally distributed among them.

        :param engine: the engine model
        :param engine_count:
        """
        self.engine = engine
        self.engine_count = engine_count

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        if flight_points.thrust is not None:
            flight_points.thrust = flight_points.thrust / self.engine_count

        self.engine.compute_flight_points(flight_points)
        flight_points.thrust = flight_points.thrust * self.engine_count

    def compute_max_power(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        return self.engine.compute_max_power(flight_points)

    def compute_weight(self):
        return self.engine.compute_weight() * self.engine_count

    def compute_dimensions(self):
        return self.engine.compute_dimensions()

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        return self.engine.compute_drag(mach, unit_reynolds, wing_mac)
