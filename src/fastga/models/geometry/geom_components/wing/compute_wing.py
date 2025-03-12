"""
Python module for wing geometry calculation, part of the geometry component.
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

import openmdao.api as om
import fastoad.api as oad

from .constants import (
    SERVICE_WING_THICKNESS_RATIO,
    SERVICE_WING_SPAN,
    SERVICE_WING_HEIGHT,
    SERVICE_WING_L1_L4,
    SERVICE_WING_L2_L3,
    SERVICE_WING_X_LOCAL,
    SERVICE_WING_X_ABSOLUTE,
    SERVICE_WING_B50,
    SERVICE_WING_MAC,
    SERVICE_WING_SWEEP,
    SUBMODEL_WING_WET_AREA,
)
from ...constants import SERVICE_WING_GEOMETRY, SUBMODEL_WING_GEOMETRY_LEGACY


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_WING_GEOMETRY, SUBMODEL_WING_GEOMETRY_LEGACY)
class ComputeWingGeometry(om.Group):
    """Wing geometry estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "wing_toc",
            oad.RegisterSubmodel.get_submodel(SERVICE_WING_THICKNESS_RATIO),
            promotes=["*"],
        )
        self.add_subsystem(
            "wing_y", oad.RegisterSubmodel.get_submodel(SERVICE_WING_SPAN), promotes=["*"]
        )
        self.add_subsystem(
            "wing_l1l4", oad.RegisterSubmodel.get_submodel(SERVICE_WING_L1_L4), promotes=["*"]
        )
        self.add_subsystem(
            "wing_l2l3", oad.RegisterSubmodel.get_submodel(SERVICE_WING_L2_L3), promotes=["*"]
        )
        self.add_subsystem(
            "wing_z", oad.RegisterSubmodel.get_submodel(SERVICE_WING_HEIGHT), promotes=["*"]
        )
        self.add_subsystem(
            "wing_x", oad.RegisterSubmodel.get_submodel(SERVICE_WING_X_LOCAL), promotes=["*"]
        )
        self.add_subsystem(
            "wing_MAC", oad.RegisterSubmodel.get_submodel(SERVICE_WING_MAC), promotes=["*"]
        )
        self.add_subsystem(
            "wing_x_absolute",
            oad.RegisterSubmodel.get_submodel(SERVICE_WING_X_ABSOLUTE),
            promotes=["*"],
        )
        self.add_subsystem(
            "wing_sweep", oad.RegisterSubmodel.get_submodel(SERVICE_WING_SWEEP), promotes=["*"]
        )
        self.add_subsystem(
            "wing_b50", oad.RegisterSubmodel.get_submodel(SERVICE_WING_B50), promotes=["*"]
        )
        self.add_subsystem(
            "wing_wet_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_WET_AREA),
            promotes=["*"],
        )
