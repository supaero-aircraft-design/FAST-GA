"""
Python module for engine weight calculation, part of the propulsion system mass computation.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

import fastoad.api as oad
import openmdao.api as om

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader

from .constants import (
    SERVICE_INSTALLED_ENGINE_MASS,
    SUBMODEL_INSTALLED_ENGINE_MASS_LEGACY,
    SUBMODEL_INSTALLED_ENGINE_MASS_RAYMER,
)

oad.RegisterSubmodel.active_models[SERVICE_INSTALLED_ENGINE_MASS] = (
    SUBMODEL_INSTALLED_ENGINE_MASS_LEGACY
)


@oad.RegisterSubmodel(SERVICE_INSTALLED_ENGINE_MASS, SUBMODEL_INSTALLED_ENGINE_MASS_LEGACY)
class ComputeEngineWeight(om.ExplicitComponent):
    """
    Engine weight estimation calling wrapper

    Based on a statistical analysis. See :cite:`raymer:2012`, Table 15.2.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("settings:weight:propulsion:engine:k_factor", val=1.0)

        self.add_output("data:weight:propulsion:engine:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="fd")
        # Overwrites the derivatives because we know the exact value
        self.declare_partials(
            of="*", wrt="settings:weight:propulsion:engine:k_factor", method="exact"
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)

        # This should give the UNINSTALLED weight
        uninstalled_engine_weight = propulsion_model.compute_weight()
        k_b1 = inputs["settings:weight:propulsion:engine:k_factor"]

        b_1 = 1.4 * uninstalled_engine_weight

        outputs["data:weight:propulsion:engine:mass"] = b_1 * k_b1

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)

        # This should give the UNINSTALLED weight
        uninstalled_engine_weight = propulsion_model.compute_weight()

        b_1 = 1.4 * uninstalled_engine_weight

        partials[
            "data:weight:propulsion:engine:mass", "settings:weight:propulsion:engine:k_factor"
        ] = b_1


@oad.RegisterSubmodel(SERVICE_INSTALLED_ENGINE_MASS, SUBMODEL_INSTALLED_ENGINE_MASS_RAYMER)
class ComputeEngineWeightRaymer(om.ExplicitComponent):
    """
    Engine weight estimation calling wrapper

    Based on a statistical analysis. See :cite:`raymer:2012` Formula 15.52, can also be found in
    :cite:`roskampart5:1985` USAF method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("settings:weight:propulsion:engine:k_factor", val=1.0)

        self.add_output("data:weight:propulsion:engine:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="fd")
        # Overwrites the derivatives because we know the exact value
        self.declare_partials(
            of="*", wrt="settings:weight:propulsion:engine:k_factor", method="exact"
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)

        k_b1 = inputs["settings:weight:propulsion:engine:k_factor"]

        # This should give the UNINSTALLED weight in lbs !
        uninstalled_engine_weight = propulsion_model.compute_weight()

        b_1 = 2.575 * uninstalled_engine_weight**0.922

        outputs["data:weight:propulsion:engine:mass"] = b_1 * k_b1

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_model = self._engine_wrapper.get_model(inputs)

        # This should give the UNINSTALLED weight in lbs !
        uninstalled_engine_weight = propulsion_model.compute_weight()

        b_1 = 2.575 * uninstalled_engine_weight**0.922

        partials[
            "data:weight:propulsion:engine:mass", "settings:weight:propulsion:engine:k_factor"
        ] = b_1
