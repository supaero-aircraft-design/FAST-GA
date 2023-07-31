"""Computation of the systems mass."""
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
    SUBMODEL_ELECTRIC_POWER_SYSTEM_MASS,
    SUBMODEL_HYDRAULIC_POWER_SYSTEM_MASS,
    SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS,
    SUBMODEL_AVIONICS_SYSTEM_MASS,
    SUBMODEL_RECORDING_SYSTEM_MASS,
)

from ..constants import SUBMODEL_SYSTEMS_MASS
from ..c_systems import ComputeSystemMass


@oad.RegisterSubmodel(SUBMODEL_SYSTEMS_MASS, "fastga.submodel.weight.mass.systems.legacy")
class SystemsWeight(om.Group):
    """Computes mass of systems."""

    def setup(self):
        self.add_subsystem(
            "navigation_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AVIONICS_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "electric_power_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_ELECTRIC_POWER_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "hydraulic_power_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HYDRAULIC_POWER_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "life_support_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "recording_systems_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_RECORDING_SYSTEM_MASS),
            promotes=["*"],
        )
        self.add_subsystem("systems_weight_sum", ComputeSystemMass(), promotes=["*"])
