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

from openmdao.core.explicitcomponent import ExplicitComponent

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from .constants import SUBMODEL_INSTALLED_ENGINE_MASS
import numpy as np

oad.RegisterSubmodel.active_models[
    SUBMODEL_INSTALLED_ENGINE_MASS
] = "fastga.submodel.weight.mass.propulsion.installed_engine.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_INSTALLED_ENGINE_MASS, "fastga.submodel.weight.mass.propulsion.installed_engine.legacy"
)
class ComputeEngineWeight(ExplicitComponent):
    """
    Engine weight estimation calling wrapper

    Based on a statistical analysis. See :cite:`raymer:2012`, Table 15.2.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("settings:weight:propulsion:engine:k_factor", val=1.0)

        self.add_output("data:weight:propulsion:engine:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = self._engine_wrapper.get_model(inputs)

        # This should give the UNINSTALLED weight
        uninstalled_engine_weight = propulsion_model.compute_weight()
        k_b1 = inputs["settings:weight:propulsion:engine:k_factor"]

        b1 = 1.4 * uninstalled_engine_weight

        outputs["data:weight:propulsion:engine:mass"] = b1 * k_b1


@oad.RegisterSubmodel(
    SUBMODEL_INSTALLED_ENGINE_MASS, "fastga.submodel.weight.mass.propulsion.installed_engine.raymer"
)
class ComputeEngineWeightRaymer(ExplicitComponent):
    """
    Engine weight estimation calling wrapper

    Based on a statistical analysis. See :cite:`raymer:2012` Formula 15.52, can also be found in
    :cite:`roskampart5:1985` USAF method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("settings:weight:propulsion:engine:k_factor", val=1.0)

        self.add_output("data:weight:propulsion:engine:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = self._engine_wrapper.get_model(inputs)

        k_b1 = inputs["settings:weight:propulsion:engine:k_factor"]

        # This should give the UNINSTALLED weight in lbs !
        uninstalled_engine_weight = propulsion_model.compute_weight()

        b1 = 2.575 * uninstalled_engine_weight ** 0.922

        outputs["data:weight:propulsion:engine:mass"] = b1 * k_b1
