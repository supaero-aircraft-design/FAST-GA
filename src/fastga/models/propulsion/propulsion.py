"""Base module for propulsion models."""
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

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import openmdao.api as om
import pandas as pd

import fastoad.api as oad
from fastoad.model_base.propulsion import IPropulsion


class IPropulsionCS23(IPropulsion):
    """
    Interface that should be implemented by propulsion models.
    """

    @abstractmethod
    def compute_weight(self) -> float:
        """
        Computes total propulsion mass.

        :return: the total uninstalled mass in kg
        """

    @abstractmethod
    def compute_max_power(self, flight_points: Union[oad.FlightPoint, pd.DataFrame]):
        """
        Computes max available power on one engine.

        :return: the maximum available power in W
        """

    @abstractmethod
    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions.

        :return: (height, width, length, wet area) of nacelle in m or m²
        """

    @abstractmethod
    def compute_drag(
        self,
        mach: Union[float, np.array],
        unit_reynolds: Union[float, np.array],
        wing_mac: float,
    ) -> Union[float, np.array]:
        """
        Computes nacelle drag force for out of fuselage engine.

        :param mach: mach at which drag should be calculated
        :param unit_reynolds: unitary Reynolds for calculation
        :param wing_mac: wing MAC length in m
        :return: drag force cd0*wing_area
        """


class BaseOMPropulsionComponent(om.ExplicitComponent, ABC):
    """
    Base class for OpenMDAO wrapping of subclasses of :class:`IEngineForOpenMDAO`.

    Classes that implements this interface should add their own inputs in setup()
    and implement :meth:`get_wrapper`.
    """

    def initialize(self):
        self.options.declare("flight_point_count", 1, types=(int, tuple))

    def setup(self):
        shape = self.options["flight_point_count"]
        self.add_input("data:propulsion:mach", np.nan, shape=shape)
        self.add_input("data:propulsion:altitude", np.nan, shape=shape, units="m")
        self.add_input("data:propulsion:engine_setting", np.nan, shape=shape)
        self.add_input("data:propulsion:use_thrust_rate", np.nan, shape=shape)
        self.add_input("data:propulsion:required_thrust_rate", np.nan, shape=shape)
        self.add_input("data:propulsion:required_thrust", np.nan, shape=shape, units="N")

        self.add_output("data:propulsion:SFC", shape=shape, units="kg/s/N")
        self.add_output("data:propulsion:thrust_rate", shape=shape)
        self.add_output("data:propulsion:thrust", shape=shape, units="N")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wrapper = self.get_wrapper().get_model(inputs)
        flight_point = oad.FlightPoint(
            mach=inputs["data:propulsion:mach"],
            altitude=inputs["data:propulsion:altitude"],
            engine_setting=inputs["data:propulsion:engine_setting"],
            thrust_is_regulated=np.logical_not(
                inputs["data:propulsion:use_thrust_rate"].astype(int)
            ),
            thrust_rate=inputs["data:propulsion:required_thrust_rate"],
            thrust=inputs["data:propulsion:required_thrust"],
        )
        wrapper.compute_flight_points(flight_point)
        outputs["data:propulsion:SFC"] = flight_point.sfc
        outputs["data:propulsion:thrust_rate"] = flight_point.thrust_rate
        outputs["data:propulsion:thrust"] = flight_point.thrust

    @staticmethod
    @abstractmethod
    def get_wrapper() -> oad.IOMPropulsionWrapper:
        """
        This method defines the used :class:`IOMPropulsionWrapper` instance.

        :return: an instance of OpenMDAO wrapper for propulsion model
        """
