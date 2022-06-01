"""
    Estimation of tail center(s) of gravity.
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

import math

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_TAIL_CG


@oad.RegisterSubmodel(SUBMODEL_TAIL_CG, "fastga.submodel.weight.cg.airframe.tail.legacy")
class ComputeTailCG(om.Group):
    def setup(self):
        self.add_subsystem("compute_ht", ComputeHTcg(), promotes=["*"])
        self.add_subsystem("compute_vt", ComputeVTcg(), promotes=["*"])


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

        self.declare_partials("*", "*", method="fd")

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
        x_cg_ht = 0.38 * b_h * math.tan(sweep_25_ht / 180.0 * math.pi) + 0.42 * l_cg
        x_cg_31 = lp_ht + fa_length - 0.25 * mac_ht + (x_cg_ht - x0_ht)

        outputs["data:weight:airframe:horizontal_tail:CG:x"] = x_cg_31


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

        self.declare_partials("*", "*", method="fd")

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
            x_cg_vt = 0.55 * b_v * math.tan(sweep_25_vt / 180.0 * math.pi) + 0.42 * l_cg_vt

        else:
            l_cg_vt = (1.0 - 0.38) * (root_chord - tip_chord) + tip_chord
            x_cg_vt = 0.38 * b_v * math.tan(sweep_25_vt / 180.0 * math.pi) + 0.42 * l_cg_vt

        x_cg_32 = lp_vt + fa_length - 0.25 * mac_vt + (x_cg_vt - x0_vt)

        outputs["data:weight:airframe:vertical_tail:CG:x"] = x_cg_32
