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

import numpy as np
import openmdao.api as om
from fastoad.constants import EngineSetting

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet


class ThrustTaxi(om.ExplicitComponent):
    """Computes the fuel consumed during the taxi phases."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:mission:sizing:taxi_out:thrust_rate", np.nan)
        self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_out:thrust", 1500, units="N")

        self.add_input("data:mission:sizing:taxi_in:thrust_rate", np.nan)
        self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_in:thrust", 1500, units="N")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )

        thrust_rate_to = inputs["data:mission:sizing:taxi_out:thrust_rate"]
        mach_to = inputs["data:mission:sizing:taxi_out:speed"] / Atmosphere(0.0).speed_of_sound

        thrust_rate_ti = inputs["data:mission:sizing:taxi_in:thrust_rate"]
        mach_ti = inputs["data:mission:sizing:taxi_in:speed"] / Atmosphere(0.0).speed_of_sound

        # FIXME: no specific settings for taxi (to be changed in fastoad\constants.py)
        flight_point_to = oad.FlightPoint(
            mach=mach_to,
            altitude=0.0,
            engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=thrust_rate_to,
        )
        propulsion_model.compute_flight_points(flight_point_to)
        thrust_to = flight_point_to.thrust

        flight_point_ti = oad.FlightPoint(
            mach=mach_ti,
            altitude=0.0,
            engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=thrust_rate_ti,
        )
        propulsion_model.compute_flight_points(flight_point_ti)
        thrust_ti = flight_point_ti.thrust

        outputs["data:mission:sizing:taxi_out:thrust"] = thrust_to
        outputs["data:mission:sizing:taxi_in:thrust"] = thrust_ti
