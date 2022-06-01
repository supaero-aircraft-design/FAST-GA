"""Estimation of engine and associated component weight."""
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

from scipy.constants import lbf
from openmdao.core.explicitcomponent import ExplicitComponent

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting


class ComputeOilWeight(ExplicitComponent):
    """
    Weight estimation for motor oil.

    Based on a statistical analysis. See :cite:`wells:2017`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_output("data:weight:propulsion:engine_oil:mass", units="lb")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        n_eng = inputs["data:geometry:propulsion:engine:count"]

        propulsion_model = self._engine_wrapper.get_model(inputs)

        flight_point = oad.FlightPoint(
            mach=0.0, altitude=0.0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=1.0
        )  # with engine_setting as EngineSetting
        propulsion_model.compute_flight_points(flight_point)

        # This should give the UNINSTALLED weight
        sl_thrust_newton = float(flight_point.thrust)
        sl_thrust_lbs = sl_thrust_newton / lbf

        b1_2 = 0.082 * n_eng * sl_thrust_lbs ** 0.65

        outputs["data:weight:propulsion:engine_oil:mass"] = b1_2
