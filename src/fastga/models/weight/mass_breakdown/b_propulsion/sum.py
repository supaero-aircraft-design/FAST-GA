"""Computation of the propulsion system mass."""
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

import openmdao.api as om

from fastoad.module_management.service_registry import RegisterSubmodel

from .b1_engine_weight import ComputeEngineWeight
from .b3_unusable_fuel_weight import ComputeUnusableFuelWeight
from .b2_fuel_lines_weight import ComputeFuelLinesWeight
from ..constants import SUBMODEL_PROPULSION_MASS


@RegisterSubmodel(SUBMODEL_PROPULSION_MASS, "fastga.submodel.weight.mass.propulsion.legacy.fuel")
class PropulsionWeight(om.Group):
    """
    Computes mass of propulsion system.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "engine_weight",
            ComputeEngineWeight(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "unusable_fuel",
            ComputeUnusableFuelWeight(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
        self.add_subsystem("fuel_lines_weight", ComputeFuelLinesWeight(), promotes=["*"])

        weight_sum = om.AddSubtractComp()
        weight_sum.add_equation(
            "data:weight:propulsion:mass",
            ["data:weight:propulsion:engine:mass", "data:weight:propulsion:fuel_lines:mass"],
            units="kg",
            desc="Mass of the propulsion system",
        )

        self.add_subsystem("airframe_weight_sum", weight_sum, promotes=["*"])
