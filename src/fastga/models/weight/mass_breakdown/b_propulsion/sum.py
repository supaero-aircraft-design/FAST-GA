"""
Python module for propulsion system mass calculation,
part of the Operating Empty Weight (OEW) estimation.
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

from .constants import (
    SERVICE_INSTALLED_ENGINE_MASS,
    SERVICE_FUEL_SYSTEM_MASS,
    SERVICE_UNUSABLE_FUEL_MASS,
)
from ..constants import SERVICE_PROPULSION_MASS, SUBMODEL_PROPULSION_MASS_LEGACY


@oad.RegisterSubmodel(SERVICE_PROPULSION_MASS, SUBMODEL_PROPULSION_MASS_LEGACY)
class PropulsionWeight(om.Group):
    """Computes mass of propulsion system."""

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "engine_weight",
            oad.RegisterSubmodel.get_submodel(
                SERVICE_INSTALLED_ENGINE_MASS, options=propulsion_option
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "unusable_fuel",
            oad.RegisterSubmodel.get_submodel(
                SERVICE_UNUSABLE_FUEL_MASS, options=propulsion_option
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "fuel_lines_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_FUEL_SYSTEM_MASS),
            promotes=["*"],
        )

        weight_sum = om.AddSubtractComp()
        weight_sum.add_equation(
            "data:weight:propulsion:mass",
            ["data:weight:propulsion:engine:mass", "data:weight:propulsion:fuel_lines:mass"],
            units="kg",
            desc="Mass of the propulsion system",
        )

        self.add_subsystem("propulsion_weight_sum", weight_sum, promotes=["*"])
