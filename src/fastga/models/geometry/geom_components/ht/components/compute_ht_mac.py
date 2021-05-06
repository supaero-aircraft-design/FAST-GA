"""
    Estimation of horizontal tail mean aerodynamic chords
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
class ComputeHTmacFD(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Horizontal tail mean aerodynamic chord estimation based on (F)ixed tail (D)istance
    """

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:length", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:y", units="m")

        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:length",
            ["data:geometry:horizontal_tail:root:chord", "data:geometry:horizontal_tail:tip:chord"],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local",
            [
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:tip:chord",
                "data:geometry:horizontal_tail:sweep_25",
                "data:geometry:horizontal_tail:span",
            ],
            method="fd",
        )
        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:y",
            [
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:tip:chord",
                "data:geometry:horizontal_tail:span",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        tmp = (
            root_chord * 0.25 + b_h / 2 * math.tan(sweep_25_ht / 180.0 * math.pi) - tip_chord * 0.25
        )

        mac_ht = (
            (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
            / (tip_chord + root_chord)
            * 2
            / 3
        )
        x0_ht = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))
        y0_ht = (b_h * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:horizontal_tail:MAC:length"] = mac_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"] = x0_ht
        outputs["data:geometry:horizontal_tail:MAC:y"] = y0_ht


class ComputeHTmacFL(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Horizontal tail mean aerodynamic chord estimation based on (F)ixed fuselage (L)ength (HTP distance computed)
    """

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:tip:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:MAC:length", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:y", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]
        fus_length = inputs["data:geometry:fuselage:length"]
        tail_type = inputs["data:geometry:has_T_tail"]
        x_vt_tip = inputs["data:geometry:vertical_tail:tip:x"]
        x_wing25 = inputs["data:geometry:wing:MAC:at25percent:x"]

        tmp = (
            root_chord * 0.25 + b_h / 2 * math.tan(sweep_25_ht / 180.0 * math.pi) - tip_chord * 0.25
        )

        mac_ht = (
            (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
            / (tip_chord + root_chord)
            * 2
            / 3
        )
        x0_ht = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))
        y0_ht = (b_h * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        if tail_type == 1.0:
            ht_lp = (x_vt_tip + x0_ht) - x_wing25
        else:
            ht_lp = (fus_length - root_chord + x0_ht) - x_wing25

        outputs["data:geometry:horizontal_tail:MAC:length"] = mac_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"] = x0_ht
        outputs["data:geometry:horizontal_tail:MAC:y"] = y0_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = ht_lp
