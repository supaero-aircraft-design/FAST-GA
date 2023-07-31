"""
    Estimation of horizontal tail mean aerodynamic chords length.
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

from ..constants import SUBMODEL_HT_MAC_LENGTH


@oad.RegisterSubmodel(
    SUBMODEL_HT_MAC_LENGTH, "fastga.submodel.geometry.horizontal_tail.mac.length.legacy"
)
class ComputeHTMacLength(om.ExplicitComponent):
    """
    Compute MAC length of the horizontal tail.
    """

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:length", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]

        mac_ht = (
            (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
            / (tip_chord + root_chord)
            * 2
            / 3
        )

        outputs["data:geometry:horizontal_tail:MAC:length"] = mac_ht

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]

        partials[
            "data:geometry:horizontal_tail:MAC:length", "data:geometry:horizontal_tail:root:chord"
        ] = (2.0 / 3.0 * (1.0 - tip_chord ** 2.0 / (root_chord + tip_chord) ** 2.0))
        partials[
            "data:geometry:horizontal_tail:MAC:length", "data:geometry:horizontal_tail:tip:chord"
        ] = (2.0 / 3.0 * (1.0 - root_chord ** 2.0 / (root_chord + tip_chord) ** 2.0))
