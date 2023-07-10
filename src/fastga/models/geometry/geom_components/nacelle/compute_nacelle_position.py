"""Estimation of nacelle and pylon geometry."""
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


import fastoad.api as oad

from ...constants import (
    SUBMODEL_NACELLE_X_POSITION,
    SUBMODEL_NACELLE_Y_POSITION,
    SUBMODEL_NACELLE_POSITION,
)

import openmdao.api as om


@oad.RegisterSubmodel(SUBMODEL_NACELLE_POSITION, "fastga.submodel.geometry.nacelle.position.legacy")
class ComputeNacellePosition(om.Group):
    # TODO: Document equations. Cite sources
    """Nacelle and pylon geometry estimation."""

    def setup(self):

        self.add_subsystem(
            "comp_y_nacelle",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_NACELLE_X_POSITION),
            promotes=["*"],
        )
        self.add_subsystem(
            "comp_x_nacelle",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_NACELLE_Y_POSITION),
            promotes=["*"],
        )
