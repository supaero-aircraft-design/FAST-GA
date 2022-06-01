"""Computation of wing area and wing area related constraints."""
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
import logging

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .wing_area_component.update_wing_area import UpdateWingArea

from .constants import (
    SUBMODEL_WING_AREA_GEOM_CONS,
    SUBMODEL_WING_AREA_GEOM_LOOP,
    SUBMODEL_WING_AREA_AERO_CONS,
    SUBMODEL_WING_AREA_AERO_LOOP,
)

_LOGGER = logging.getLogger(__name__)


@oad.RegisterOpenMDAOSystem("fastga.loop.wing_area", domain=ModelDomain.OTHER)
class UpdateWingAreaGroup(om.Group):
    """
    Groups that gather the computation of the updated wing area, chooses the biggest one and
    computes the constraints based on the new wing area.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):
        """Adding the update groups, the selection of the maximum and the constraints."""
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "loop_wing_area_geom",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_AREA_GEOM_LOOP),
            promotes_inputs=["*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "loop_wing_area_aero",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_WING_AREA_AERO_LOOP, options=propulsion_option
            ),
            promotes_inputs=["*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "update_wing_area",
            UpdateWingArea(),
            promotes_inputs=[],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "constraint_wing_area_geom",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_AREA_GEOM_CONS),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        self.add_subsystem(
            "constraint_wing_area_aero",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_WING_AREA_AERO_CONS, options=propulsion_option
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.connect("loop_wing_area_geom.wing_area", "update_wing_area.wing_area:geometric")
        self.connect("loop_wing_area_aero.wing_area", "update_wing_area.wing_area:aerodynamic")
