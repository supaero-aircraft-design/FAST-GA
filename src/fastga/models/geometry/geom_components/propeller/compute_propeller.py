"""Estimation of propeller position wrt to the wing."""
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

from ...constants import SUBMODEL_PROPELLER_GEOMETRY
from .constants import SUBMODEL_PROPELLER_POSITION, SUBMODEL_PROPELLER_INSTALLATION


@oad.RegisterSubmodel(
    SUBMODEL_PROPELLER_GEOMETRY, "fastga.submodel.geometry.propeller.geometry.legacy"
)
class ComputePropellerGeometry(om.Group):
    """Propeller position with respect to the leading edge estimation."""

    def setup(self):

        self.add_subsystem(
            "propeller_position",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PROPELLER_POSITION),
            promotes=["*"],
        )
        self.add_subsystem(
            "propeller_effective_advance_ratio",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PROPELLER_INSTALLATION),
            promotes=["*"],
        )
