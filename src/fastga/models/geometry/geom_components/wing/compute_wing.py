"""Estimation of wing geometry."""
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

from openmdao.api import Group

import fastoad.api as oad

from .constants import (
    SUBMODEL_WING_THICKNESS_RATIO,
    SUBMODEL_WING_SPAN,
    SUBMODEL_WING_L1_L4,
    SUBMODEL_WING_L2_L3,
    SUBMODEL_WING_X_LOCAL,
    SUBMODEL_WING_B50,
    SUBMODEL_WING_MAC,
    SUBMODEL_WING_SWEEP,
    SUBMODEL_WING_WET_AREA,
)
from ...constants import SUBMODEL_WING_GEOMETRY


@oad.RegisterSubmodel(SUBMODEL_WING_GEOMETRY, "fastga.submodel.geometry.wing.legacy")
class ComputeWingGeometry(Group):
    # TODO: Document equations. Cite sources
    """Wing geometry estimation."""

    def setup(self):
        self.add_subsystem(
            "wing_toc",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_THICKNESS_RATIO),
            promotes=["*"],
        )
        self.add_subsystem(
            "wing_y", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_SPAN), promotes=["*"]
        )
        self.add_subsystem(
            "wing_l1l4", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_L1_L4), promotes=["*"]
        )
        self.add_subsystem(
            "wing_l2l3", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_L2_L3), promotes=["*"]
        )
        self.add_subsystem(
            "wing_x", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_X_LOCAL), promotes=["*"]
        )
        self.add_subsystem(
            "wing_b50", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_B50), promotes=["*"]
        )
        self.add_subsystem(
            "wing_mac", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_MAC), promotes=["*"]
        )
        self.add_subsystem(
            "wing_sweep", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_SWEEP), promotes=["*"]
        )
        self.add_subsystem(
            "wing_wet_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_WET_AREA),
            promotes=["*"],
        )
