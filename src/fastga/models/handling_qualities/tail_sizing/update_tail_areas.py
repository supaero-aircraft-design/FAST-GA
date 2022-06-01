"""Computation of tail areas w.r.t. HQ criteria."""
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
from fastoad.module_management.constants import ModelDomain

from .constants import SUBMODEL_HT_AREA, SUBMODEL_VT_AREA


@oad.RegisterOpenMDAOSystem(
    "fastga.handling_qualities.tail_sizing", domain=ModelDomain.HANDLING_QUALITIES
)
class UpdateTailAreas(om.Group):
    """
    Computes areas of vertical and horizontal tail.

    - Horizontal tail area is computed so it can balance pitching moment of
      aircraft at rotation speed.
    - Vertical tail area is computed so aircraft can have the Cnbeta in cruise
      conditions and (for bi-motor) maintain trajectory with failed engine @ 5000ft.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "horizontal_tail",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_AREA, options=propulsion_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "vertical_tail",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_AREA, options=propulsion_option),
            promotes=["*"],
        )
