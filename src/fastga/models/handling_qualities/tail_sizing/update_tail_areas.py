"""Computation of tail areas w.r.t. HQ criteria."""
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

import numpy as np
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


@oad.RegisterOpenMDAOSystem(
    "fastga.handling_qualities.tail_sizing.volumetric", domain=ModelDomain.HANDLING_QUALITIES
)
class UpdateTailAreasVolumetric(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:volumetric_coefficient", val=np.nan)
        self.add_input("data:geometry:vertical_tail:volumetric_coefficient", val=np.nan)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )

        self.add_output("data:geometry:horizontal_tail:area", val=4.0, units="m**2")
        self.add_output("data:geometry:vertical_tail:area", val=4.0, units="m**2")

    def setup_partials(self):
        self.declare_partials(
            of="data:geometry:horizontal_tail:area",
            wrt=[
                "data:geometry:wing:area",
                "data:geometry:wing:MAC:length",
                "data:geometry:horizontal_tail:volumetric_coefficient",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ],
        )
        self.declare_partials(
            of="data:geometry:vertical_tail:area",
            wrt=[
                "data:geometry:wing:area",
                "data:geometry:wing:span",
                "data:geometry:vertical_tail:volumetric_coefficient",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_wing = inputs["data:geometry:wing:area"]
        b_wing = inputs["data:geometry:wing:span"]
        mac_wing = inputs["data:geometry:wing:MAC:length"]
        l_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        vc_ht = inputs["data:geometry:horizontal_tail:volumetric_coefficient"]
        vc_vt = inputs["data:geometry:vertical_tail:volumetric_coefficient"]

        outputs["data:geometry:horizontal_tail:area"] = vc_ht * s_wing * mac_wing / l_ht
        outputs["data:geometry:vertical_tail:area"] = vc_vt * s_wing * b_wing / l_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        s_wing = inputs["data:geometry:wing:area"]
        b_wing = inputs["data:geometry:wing:span"]
        mac_wing = inputs["data:geometry:wing:MAC:length"]
        l_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        vc_ht = inputs["data:geometry:horizontal_tail:volumetric_coefficient"]
        vc_vt = inputs["data:geometry:vertical_tail:volumetric_coefficient"]

        partials["data:geometry:horizontal_tail:area", "data:geometry:wing:area"] = (
                vc_ht * mac_wing / l_ht
        )

        partials["data:geometry:horizontal_tail:area", "data:geometry:wing:MAC:length"] = (
                vc_ht * s_wing / l_ht
        )

        partials[
            "data:geometry:horizontal_tail:area",
            "data:geometry:horizontal_tail:volumetric_coefficient",
        ] = s_wing * mac_wing / l_ht

        partials[
            "data:geometry:horizontal_tail:area",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = -vc_ht * s_wing * mac_wing / l_ht ** 2.0

        partials["data:geometry:vertical_tail:area", "data:geometry:wing:area"] = (
                vc_vt * b_wing / l_vt
        )

        partials["data:geometry:vertical_tail:area", "data:geometry:wing:span"] = (
                vc_vt * s_wing / l_vt
        )

        partials[
            "data:geometry:vertical_tail:area", "data:geometry:vertical_tail:volumetric_coefficient"
        ] = s_wing * b_wing / l_vt

        partials[
            "data:geometry:vertical_tail:area",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = -vc_vt * s_wing * b_wing / l_vt ** 2.0
