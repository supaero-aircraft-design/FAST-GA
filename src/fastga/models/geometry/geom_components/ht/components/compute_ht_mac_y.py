"""
    Estimation of horizontal tail mean aerodynamic chords y position.
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

from ..constants import SUBMODEL_HT_MAC_Y


@oad.RegisterSubmodel(SUBMODEL_HT_MAC_Y, "fastga.submodel.geometry.horizontal_tail.mac.y.legacy")
class ComputeHTMacY(om.ExplicitComponent):
    """
    Compute y coordinate of the horizontal tail's MAC.
    """

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:y", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        y0_ht = (b_h * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:horizontal_tail:MAC:y"] = y0_ht

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        partials["data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:span"] = (
            0.5 * root_chord + tip_chord
        ) / (3 * (root_chord + tip_chord))
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:root:chord"
        ] = (-b_h * tip_chord / (6.0 * (root_chord + tip_chord) ** 2.0))
        partials[
            "data:geometry:horizontal_tail:MAC:y", "data:geometry:horizontal_tail:tip:chord"
        ] = (b_h * root_chord / (6.0 * (root_chord + tip_chord) ** 2.0))
