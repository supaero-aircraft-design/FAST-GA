"""Simple module for taxi computation."""
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
import numpy as np
import openmdao.api as om

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting

from stdatm import Atmosphere

from ..constants import SUBMODEL_TAXI

_LOGGER = logging.getLogger(__name__)

oad.RegisterSubmodel.active_models[
    SUBMODEL_TAXI
] = "fastga.submodel.performances.mission.taxi.legacy"


@oad.RegisterSubmodel(SUBMODEL_TAXI, "fastga.submodel.performances.mission.taxi.legacy")
class ComputeTaxi(om.ExplicitComponent):
    """Compute the fuel consumption for taxi based on speed and duration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("taxi_out", default=True, types=bool)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        if self.options["taxi_out"]:
            self.add_input("data:mission:sizing:taxi_out:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
            self.add_output("data:mission:sizing:taxi_out:fuel", units="kg")
        else:
            self.add_input("data:mission:sizing:taxi_in:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_in:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
            self.add_output("data:mission:sizing:taxi_in:fuel", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["taxi_out"]:
            _LOGGER.info("Entering mission computation")

        propulsion_model = self._engine_wrapper.get_model(inputs)
        if self.options["taxi_out"]:
            thrust_rate = inputs["data:mission:sizing:taxi_out:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_out:duration"]
            mach = inputs["data:mission:sizing:taxi_out:speed"] / Atmosphere(0.0).speed_of_sound
        else:
            thrust_rate = inputs["data:mission:sizing:taxi_in:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_in:duration"]
            mach = inputs["data:mission:sizing:taxi_in:speed"] / Atmosphere(0.0).speed_of_sound

        # FIXME: no specific settings for taxi (to be changed in fastoad\constants.py)
        flight_point = oad.FlightPoint(
            mach=mach, altitude=0.0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=thrust_rate
        )
        propulsion_model.compute_flight_points(flight_point)
        fuel_mass = propulsion_model.get_consumed_mass(flight_point, duration)

        if self.options["taxi_out"]:
            outputs["data:mission:sizing:taxi_out:fuel"] = fuel_mass
        else:
            outputs["data:mission:sizing:taxi_in:fuel"] = fuel_mass
