"""Estimation of wing Zs (sections height)."""
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

import numpy as np

import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_WING_HEIGHT


@oad.RegisterSubmodel(SUBMODEL_WING_HEIGHT, "fastga.submodel.geometry.wing.height.legacy")
class ComputeWingZ(om.ExplicitComponent):
    """
    Computation of the distance between the fuselage center line and the wing. Based on simple
    geometric considerations.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)

        self.add_output(
            "data:geometry:wing:root:z",
            units="m",
            desc="Distance from the fuselage center line to the middle of the wing, taken positive "
            "when wing is below fuselage center line",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # TODO: Add dihedral and high/mid wing configurations

        h_f = inputs["data:geometry:fuselage:maximum_height"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        thickness_ratio_2_wing = inputs["data:geometry:wing:root:thickness_ratio"]

        z2_wing = (h_f - thickness_ratio_2_wing * l2_wing) * 0.5

        outputs["data:geometry:wing:root:z"] = z2_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        l2_wing = inputs["data:geometry:wing:root:chord"]
        thickness_ratio_2_wing = inputs["data:geometry:wing:root:thickness_ratio"]

        partials["data:geometry:wing:root:z", "data:geometry:fuselage:maximum_height"] = 0.5
        partials["data:geometry:wing:root:z", "data:geometry:wing:root:chord"] = (
            -0.5 * thickness_ratio_2_wing
        )
        partials["data:geometry:wing:root:z", "data:geometry:wing:root:thickness_ratio"] = (
            -0.5 * l2_wing
        )
