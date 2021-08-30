"""
Estimation of engine and associated component weight
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

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet


class ComputeEngineWeight(ExplicitComponent):
    """
    Engine weight estimation calling wrapper

    Based on a statistical analysis. See :cite:`raymer:2012`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", val=np.nan)

        self.add_output("data:weight:propulsion:engine:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )

        # This should give the UNINSTALLED weight
        uninstalled_engine_weight = propulsion_model.compute_weight()

        b1 = 1.4 * uninstalled_engine_weight

        outputs["data:weight:propulsion:engine:mass"] = b1
