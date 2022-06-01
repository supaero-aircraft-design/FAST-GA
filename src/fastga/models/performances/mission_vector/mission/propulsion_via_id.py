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

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from stdatm import Atmosphere

from fastga.models.performances.mission_vector.constants import SUBMODEL_ENERGY_CONSUMPTION

oad.RegisterSubmodel.active_models[
    SUBMODEL_ENERGY_CONSUMPTION
] = "fastga.submodel.performances.energy_consumption.ICE"


@oad.RegisterSubmodel(
    SUBMODEL_ENERGY_CONSUMPTION, "fastga.submodel.performances.energy_consumption.ICE"
)
class FuelConsumed(om.ExplicitComponent):
    """Computes the fuel consumed at each time step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "thrust_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="N",
        )
        self.add_input(
            "altitude_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="m",
        )
        self.add_input(
            "time_step_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="s",
        )
        self.add_input(
            "true_airspeed_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="m/s",
        )
        self.add_input(
            "engine_setting_econ", shape=number_of_points + 2, val=np.full(number_of_points + 2, 1)
        )

        self.add_output(
            "fuel_consumed_t_econ",
            val=np.full(number_of_points + 2, 0.0),
            desc="fuel consumed at each time step",
            units="kg",
        )
        self.add_output(
            "non_consumable_energy_t_econ",
            val=np.full(number_of_points + 2, 0.0),
            desc="fuel consumed at each time step",
            units="W*h",
        )
        self.add_output(
            "thrust_rate_t_econ",
            val=np.full(number_of_points + 2, 0.5),
            desc="thrust ratio at each time step",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = self._engine_wrapper.get_model(inputs)

        time_step = inputs["time_step_econ"]
        atm = Atmosphere(inputs["altitude_econ"], altitude_in_feet=False)
        atm.true_airspeed = inputs["true_airspeed_econ"]

        engine_setting = inputs["engine_setting_econ"]

        # TODO : Change the EngineSetting based on the phase we are in
        flight_point = oad.FlightPoint(
            mach=atm.mach,
            altitude=inputs["altitude_econ"],
            engine_setting=engine_setting,
            thrust_is_regulated=np.full_like(inputs["altitude_econ"], True),
            thrust=inputs["thrust_econ"],
            thrust_rate=np.full_like(inputs["altitude_econ"], 0.0),
        )
        propulsion_model.compute_flight_points(flight_point)

        consumed_mass_1s = propulsion_model.get_consumed_mass(flight_point, 1.0)
        fuel_consumed_t = consumed_mass_1s * time_step

        outputs["fuel_consumed_t_econ"] = fuel_consumed_t
        outputs["thrust_rate_t_econ"] = flight_point.thrust_rate
