"""
    Estimation of horizontal tail's center of gravity.
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

from ..constants import SUBMODEL_HORIZONTAL_TAIL_CG


@oad.RegisterSubmodel(
    SUBMODEL_HORIZONTAL_TAIL_CG, "fastga.submodel.weight.cg.airframe.horizontal_tail.legacy"
)
class ComputeHTcg(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Horizontal tail center of gravity estimation

    Based on a statistical analysis. See :cite:`roskampart5:1985`.
    """

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m"
        )

        self.add_output("data:weight:airframe:horizontal_tail:CG:x", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        x0_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]

        l_cg = (1.0 - 0.38) * (root_chord - tip_chord) + tip_chord
        x_cg_ht = 0.38 * b_h * np.tan(sweep_25_ht / 180.0 * np.pi) + 0.42 * l_cg
        x_cg_31 = lp_ht + fa_length - 0.25 * mac_ht + (x_cg_ht - x0_ht)

        outputs["data:weight:airframe:horizontal_tail:CG:x"] = x_cg_31

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        b_h = inputs["data:geometry:horizontal_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]

        partials[
            "data:weight:airframe:horizontal_tail:CG:x", "data:geometry:horizontal_tail:root:chord"
        ] = 0.2604
        partials[
            "data:weight:airframe:horizontal_tail:CG:x", "data:geometry:horizontal_tail:tip:chord"
        ] = 0.1596
        partials[
            "data:weight:airframe:horizontal_tail:CG:x", "data:geometry:horizontal_tail:span"
        ] = 0.38 * np.tan((np.pi * sweep_25_ht) / 180)
        partials[
            "data:weight:airframe:horizontal_tail:CG:x", "data:geometry:horizontal_tail:sweep_25"
        ] = (19 / 9000 * b_h * np.pi * (np.tan((np.pi * sweep_25_ht) / 180) ** 2 + 1))
        partials[
            "data:weight:airframe:horizontal_tail:CG:x", "data:geometry:wing:MAC:at25percent:x"
        ] = 1.0
        partials[
            "data:weight:airframe:horizontal_tail:CG:x",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = 1.0
        partials[
            "data:weight:airframe:horizontal_tail:CG:x", "data:geometry:horizontal_tail:MAC:length"
        ] = -0.25
        partials[
            "data:weight:airframe:horizontal_tail:CG:x",
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
        ] = -1.0
