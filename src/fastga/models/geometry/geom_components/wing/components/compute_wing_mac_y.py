"""Estimation of wing mean aerodynamic chord y coordinate"""
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

from ..constants import SUBMODEL_WING_MAC_Y


@oad.RegisterSubmodel(SUBMODEL_WING_MAC_Y, "fastga.submodel.geometry.wing.mac.y")
class ComputeWingMacY(om.ExplicitComponent):
    """
    Compute y coordinate of the wing's MAC.
    """

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:MAC:y", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        y0_wing = (
            3 * y2_wing ** 2 * l2_wing
            + (y4_wing - y2_wing)
            * (l4_wing * (y2_wing + 2 * y4_wing) + l2_wing * (y4_wing + 2 * y2_wing))
        ) / (3 * wing_area)

        outputs["data:geometry:wing:MAC:y"] = y0_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        partials["data:geometry:wing:MAC:y", "data:geometry:wing:area"] = (
            (l2_wing * (2 * y2_wing + y4_wing) + l4_wing * (y2_wing + 2 * y4_wing))
            * (y2_wing - y4_wing)
            - 3 * l2_wing * y2_wing ** 2
        ) / (3 * wing_area ** 2)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:root:y"] = -(
            l2_wing * (2 * y2_wing + y4_wing)
            - 6 * l2_wing * y2_wing
            + l4_wing * (y2_wing + 2 * y4_wing)
            + (2 * l2_wing + l4_wing) * (y2_wing - y4_wing)
        ) / (3 * wing_area)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:tip:y"] = (
            l2_wing * (2 * y2_wing + y4_wing)
            + l4_wing * (y2_wing + 2 * y4_wing)
            - (l2_wing + 2 * l4_wing) * (y2_wing - y4_wing)
        ) / (3 * wing_area)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:root:chord"] = -(
            (y2_wing - y4_wing) * (2 * y2_wing + y4_wing) - 3 * y2_wing ** 2
        ) / (3 * wing_area)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:tip:chord"] = -(
            (y2_wing - y4_wing) * (y2_wing + 2 * y4_wing)
        ) / (3 * wing_area)
