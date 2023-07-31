"""Estimation of wing MAC X absolute."""
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

from ..constants import SUBMODEL_WING_X_ABSOLUTE_MAC


@oad.RegisterSubmodel(
    SUBMODEL_WING_X_ABSOLUTE_MAC, "fastga.submodel.geometry.wing.x_absolute.mac.legacy"
)
class ComputeWingXAbsoluteMac(om.ExplicitComponent):
    """
    Wing absolute Xs estimation, distance from the nose to the MAC leading edge
    """

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", units="m", val=np.nan)
        self.add_input("data:geometry:wing:MAC:at25percent:x", units="m", val=np.nan)

        self.add_output("data:geometry:wing:MAC:leading_edge:x:absolute", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]

        x_abs_mac = fa_length - 0.25 * l0_wing

        outputs["data:geometry:wing:MAC:leading_edge:x:absolute"] = x_abs_mac

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials[
            "data:geometry:wing:MAC:leading_edge:x:absolute",
            "data:geometry:wing:MAC:at25percent:x",
        ] = 1.0
        partials[
            "data:geometry:wing:MAC:leading_edge:x:absolute",
            "data:geometry:wing:MAC:length",
        ] = -0.25
