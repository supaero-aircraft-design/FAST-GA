"""
    Estimation of vertical tail mean aerodynamic chord z position.
"""

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

from ..constants import SUBMODEL_VT_POSITION_Z


@oad.RegisterSubmodel(SUBMODEL_VT_POSITION_Z, "fastga.submodel.geometry.vertical_tail.position.z")
class ComputeVTMacZ(om.ExplicitComponent):
    """
    Compute z coordinate of the vertical tail's MAC.
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        z0_vt = (2 * b_v * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        tmp = 3 * root_chord + 3 * tip_chord

        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:root:chord"] = (
            b_v / tmp - (6 * b_v * (root_chord / 2 + tip_chord)) / tmp ** 2
        )
        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:tip:chord"] = (
            2 * b_v
        ) / tmp - (6 * b_v * (root_chord / 2 + tip_chord)) / tmp ** 2
        partials["data:geometry:vertical_tail:MAC:z", "data:geometry:vertical_tail:span"] = (
            2 * (root_chord / 2 + tip_chord)
        ) / tmp
