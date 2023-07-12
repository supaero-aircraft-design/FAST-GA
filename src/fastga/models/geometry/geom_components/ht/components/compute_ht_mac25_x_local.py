"""
    Estimation of horizontal tail 25% mean aerodynamic chords x position in local coordinate.
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

from ..constants import SUBMODEL_HT_MAC_X_LOCAL


@oad.RegisterSubmodel(
    SUBMODEL_HT_MAC_X_LOCAL, "fastga.submodel.geometry.horizontal_tail.mac.x_local.legacy"
)
class ComputeHTMacX25(om.ExplicitComponent):
    """
    Compute x coordinate (local) at 25% MAC of the horizontal tail.
    """

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        tmp = root_chord * 0.25 + b_h / 2 * np.tan(sweep_25_ht) - tip_chord * 0.25
        x0_ht = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"] = x0_ht

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        tmp = root_chord * 0.25 + b_h / 2 * np.tan(sweep_25_ht) - tip_chord * 0.25
        d_tmp_d_rc = 0.25
        d_tmp_d_tc = -0.25
        d_tmp_d_bh = 0.5 * np.tan(sweep_25_ht)
        d_tmp_d_sweep = b_h / 2 * (1.0 + np.tan(sweep_25_ht) ** 2.0)

        tmp_2 = (root_chord + 2 * tip_chord) / (3 * (root_chord + tip_chord))
        d_tmp_2_d_rc = -tip_chord / (3.0 * (root_chord + tip_chord) ** 2.0)
        d_tmp_2_d_tc = root_chord / (3.0 * (root_chord + tip_chord) ** 2.0)

        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:root:chord",
        ] = (
            d_tmp_d_rc * tmp_2 + tmp * d_tmp_2_d_rc
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:tip:chord",
        ] = (
            d_tmp_d_tc * tmp_2 + tmp * d_tmp_2_d_tc
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:sweep_25",
        ] = (
            tmp_2 * d_tmp_d_sweep
        )
        partials[
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            "data:geometry:horizontal_tail:span",
        ] = (
            tmp_2 * d_tmp_d_bh
        )
