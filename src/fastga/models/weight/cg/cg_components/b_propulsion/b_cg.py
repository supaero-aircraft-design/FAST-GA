"""Estimation of center of gravity for a fuel propulsion system."""
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

from ..constants import (
    SUBMODEL_ENGINE_CG,
    SUBMODEL_FUEL_LINES_CG,
    SUBMODEL_FUEL_PROPULSION_CG,
    SUBMODEL_PROPULSION_CG,
)


@oad.RegisterSubmodel(SUBMODEL_PROPULSION_CG, "fastga.submodel.weight.cg.propulsion.legacy")
class FuelPropulsionCG(om.Group):
    def setup(self):

        self.add_subsystem(
            "engine_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_ENGINE_CG), promotes=["*"]
        )
        self.add_subsystem(
            "fuel_lines_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUEL_LINES_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "propulsion_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUEL_PROPULSION_CG),
            promotes=["*"],
        )
