"""
    Estimation of vertical tail mean aerodynamic chords
"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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
from openmdao.core.explicitcomponent import ExplicitComponent


# TODO: it would be good to have a function to compute MAC for HT, VT and WING
class ComputeVTmacFD(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord estimation based on (F)ixed tail (D)istance
    """

    def setup(self):
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials(
            "data:geometry:vertical_tail:MAC:length",
            [
                "data:geometry:vertical_tail:root:chord",
                "data:geometry:vertical_tail:tip:chord"
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:at25percent:x:local", "*", method="fd"
        )
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:z",
            [
                "data:geometry:vertical_tail:root:chord",
                "data:geometry:vertical_tail:tip:chord",
                "data:geometry:vertical_tail:span",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            [
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        has_t_tail = inputs["data:geometry:has_T_tail"]

        tmp = root_chord * 0.25 + b_v * math.tan(sweep_25_vt / 180.0 * math.pi) - tip_chord * 0.25

        mac_vt = (
            (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
            / (tip_chord + root_chord)
            * 2.0
            / 3.0
        )
        x0_vt = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))
        z0_vt = (2 * b_v * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        if has_t_tail:
            vt_lp = lp_ht - b_v * math.tan(sweep_25_vt / 180.0 * math.pi)

        else:
            vt_lp = lp_ht

        outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt
        outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = vt_lp


class ComputeVTmacFL(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord estimation based on (F)ixed fuselage (L)ength (VTP distance computed)
    """

    def setup(self):
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:tip:x", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        fus_length = inputs["data:geometry:fuselage:length"]
        x_wing25 = inputs["data:geometry:wing:MAC:at25percent:x"]


        tmp = root_chord * 0.25 + b_v * math.tan(sweep_25_vt / 180.0 * math.pi) - tip_chord * 0.25

        mac_vt = (
                (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
                / (tip_chord + root_chord)
                * 2.0
                / 3.0
        )
        x0_vt = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))
        z0_vt = (2 * b_v * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        vt_lp = (fus_length - root_chord + x0_vt) - x_wing25
        x_tip = b_v * math.tan(sweep_25_vt / 180.0 * math.pi) + x_wing25 + (vt_lp - x0_vt)

        outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt
        outputs["data:geometry:vertical_tail:tip:x"] = x_tip
        outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = vt_lp
