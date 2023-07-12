"""Estimation of landing gears geometry."""
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

from ...constants import SUBMODEL_LANDING_GEAR_GEOMETRY

from .constants import SUBMODEL_LANDING_GEAR_HEIGHT, SUBMODEL_LANDING_GEAR_POSITION


@oad.RegisterSubmodel(
    SUBMODEL_LANDING_GEAR_GEOMETRY, "fastga.submodel.geometry.landing_gear.legacy"
)
class ComputeLandingGearsGeometry(om.Group):
    # TODO: Document equations. Cite sources
    """
    Landing gears geometry estimation. Position along the span is based on aircraft pictures
    analysis.
    """

    def setup(self):

        self.add_subsystem(
            "landing_gear_height",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LANDING_GEAR_HEIGHT),
            promotes=["*"],
        )
        self.add_subsystem(
            "landing_gear_position",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LANDING_GEAR_POSITION),
            promotes=["*"],
        )
