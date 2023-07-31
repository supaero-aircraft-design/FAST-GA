"""
Estimation of Operating Weight Empty (OWE).
"""
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

import openmdao.api as om
import fastoad.api as oad

from .constants import (
    SUBMODEL_AIRFRAME_MASS,
    SUBMODEL_PROPULSION_MASS,
    SUBMODEL_SYSTEMS_MASS,
    SUBMODEL_FURNITURE_MASS,
    SUBMODEL_OWE,
)

from .compute_owe import ComputeOWE


@oad.RegisterSubmodel(SUBMODEL_OWE, "fastga.submodel.weight.mass.owe.legacy")
class ComputeOperatingWeightEmpty(om.Group):
    """Operating Weight Empty (OWE) estimation

    This group aggregates weight from all components of the aircraft.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        # Airframe
        self.add_subsystem(
            "airframe_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRFRAME_MASS),
            promotes=["*"],
        )
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "propulsion_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PROPULSION_MASS, options=propulsion_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_SYSTEMS_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "furniture_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FURNITURE_MASS),
            promotes=["*"],
        )
        self.add_subsystem("owe_sum", ComputeOWE(), promotes=["*"])
