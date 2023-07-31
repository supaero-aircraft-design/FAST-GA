"""
    Estimation of vertical tail's center of gravity.
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

from ..constants import SUBMODEL_VERTICAL_TAIL_CG


@oad.RegisterSubmodel(
    SUBMODEL_VERTICAL_TAIL_CG, "fastga.submodel.weight.cg.airframe.vertical_tail.legacy"
)
class ComputeVTcg(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail center of gravity estimation."""

    def setup(self):

        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:weight:airframe:vertical_tail:CG:x", units="m")

        self.declare_partials(
            of="data:weight:airframe:vertical_tail:CG:x",
            wrt=[
                "data:geometry:vertical_tail:MAC:length",
                "data:geometry:vertical_tail:root:chord",
                "data:geometry:vertical_tail:tip:chord",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:vertical_tail:MAC:at25percent:x:local",
                "data:geometry:vertical_tail:sweep_25",
                "data:geometry:vertical_tail:span",
                "data:geometry:wing:MAC:at25percent:x",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        mac_vt = inputs["data:geometry:vertical_tail:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:local"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        has_t_tail = inputs["data:geometry:has_T_tail"]

        if has_t_tail:
            l_cg_vt = (1.0 - 0.55) * (root_chord - tip_chord) + tip_chord
            x_cg_vt = 0.55 * b_v * np.tan(sweep_25_vt / 180.0 * np.pi) + 0.42 * l_cg_vt

        else:
            l_cg_vt = (1.0 - 0.38) * (root_chord - tip_chord) + tip_chord
            x_cg_vt = 0.38 * b_v * np.tan(sweep_25_vt / 180.0 * np.pi) + 0.42 * l_cg_vt

        x_cg_32 = lp_vt + fa_length - 0.25 * mac_vt + (x_cg_vt - x0_vt)

        outputs["data:weight:airframe:vertical_tail:CG:x"] = x_cg_32

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        has_t_tail = inputs["data:geometry:has_T_tail"]

        if has_t_tail:
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:root:chord"
            ] = 0.189
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:tip:chord"
            ] = 0.231
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:sweep_25"
            ] = (11 / 3600 * b_v * np.pi * (np.tan((np.pi * sweep_25_vt) / 180) ** 2 + 1))
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:span"
            ] = 0.55 * np.tan((np.pi * sweep_25_vt) / 180)

        else:
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:root:chord"
            ] = 0.2604
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:tip:chord"
            ] = 0.1596
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:sweep_25"
            ] = (19 / 9000 * b_v * np.pi * (np.tan((np.pi * sweep_25_vt) / 180) ** 2 + 1))
            partials[
                "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:span"
            ] = 0.38 * np.tan((np.pi * sweep_25_vt) / 180)

        partials[
            "data:weight:airframe:vertical_tail:CG:x",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = 1.0
        partials[
            "data:weight:airframe:vertical_tail:CG:x", "data:geometry:vertical_tail:MAC:length"
        ] = -0.25
        partials[
            "data:weight:airframe:vertical_tail:CG:x", "data:geometry:wing:MAC:at25percent:x"
        ] = 1.0
        partials[
            "data:weight:airframe:vertical_tail:CG:x",
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
        ] = -1.0
