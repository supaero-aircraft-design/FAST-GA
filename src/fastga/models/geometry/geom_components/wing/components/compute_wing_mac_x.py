"""Estimation of wing mean aerodynamic chord x local coordinate"""
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

from ..constants import SUBMODEL_WING_MAC_X


@oad.RegisterSubmodel(SUBMODEL_WING_MAC_X, "fastga.submodel.geometry.wing.mac.x.legacy")
class ComputeWingMacX(om.ExplicitComponent):
    """
    Compute x coordinate (local) of the leading edge of the wing MAC.
    """

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:MAC:leading_edge:x:local", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        x0_wing = (x4_wing * ((y4_wing - y2_wing) * (2 * l4_wing + l2_wing))) / (3 * wing_area)

        outputs["data:geometry:wing:MAC:leading_edge:x:local"] = x0_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:area"] = (
            x4_wing * (l2_wing + 2 * l4_wing) * (y2_wing - y4_wing)
        ) / (3 * wing_area ** 2)
        partials[
            "data:geometry:wing:MAC:leading_edge:x:local",
            "data:geometry:wing:tip:leading_edge:x:local",
        ] = -((l2_wing + 2 * l4_wing) * (y2_wing - y4_wing)) / (3 * wing_area)
        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:root:y"] = -(
            x4_wing * (l2_wing + 2 * l4_wing)
        ) / (3 * wing_area)
        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:tip:y"] = (
            x4_wing * (l2_wing + 2 * l4_wing)
        ) / (3 * wing_area)
        partials[
            "data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:root:chord"
        ] = -(x4_wing * (y2_wing - y4_wing)) / (3 * wing_area)
        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:tip:chord"] = -(
            2 * x4_wing * (y2_wing - y4_wing)
        ) / (3 * wing_area)
