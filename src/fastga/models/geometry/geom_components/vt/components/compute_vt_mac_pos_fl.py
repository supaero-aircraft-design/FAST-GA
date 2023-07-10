"""Estimation of vertical tail mean aerodynamic chord position based on (F)ixed fuselage (L)ength."""

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
    SUBMODEL_VT_POSITION_FL,
    SUBMODEL_VT_POSITION_FL_X25_LOCAL,
    SUBMODEL_VT_POSITION_FL_X25_W,
    SUBMODEL_VT_POSITION_FL_TIP_X,
)

oad.RegisterSubmodel.active_models[
    SUBMODEL_VT_POSITION_FL
] = "fastga.submodel.geometry.vertical_tail.position.fl.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_VT_POSITION_FL, "fastga.submodel.geometry.vertical_tail.position.fl.legacy"
)
class ComputeVTMacPositionFL(om.Group):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord position estimation based on (F)ixed fuselage (L)ength (VTP
    distance computed).
    """

    def setup(self):

        self.add_subsystem(
            "comp_x_pos_local_25MAC_fl",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_POSITION_FL_X25_LOCAL),
            promotes=["*"],
        )
        self.add_subsystem(
            "comp_x_pos_from_wing_25MAC_fl",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_POSITION_FL_X25_W),
            promotes=["*"],
        )
        self.add_subsystem(
            "comp_x_pos_tip_fl",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_POSITION_FL_TIP_X),
            promotes=["*"],
        )
