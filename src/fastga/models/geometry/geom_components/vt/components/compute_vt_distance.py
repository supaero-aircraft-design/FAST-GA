"""Estimation of vertical tail mean aerodynamic chord position."""

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
from openmdao.core.explicitcomponent import ExplicitComponent
import fastoad.api as oad

from ..constants import SUBMODEL_VT_POSITION_FL, SUBMODEL_VT_POSITION_FD

oad.RegisterSubmodel.active_models[
    SUBMODEL_VT_POSITION_FD
] = "fastga.submodel.geometry.vertical_tail.position.fd.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_VT_POSITION_FD, "fastga.submodel.geometry.vertical_tail.position.fd.legacy"
)
class ComputeVTMacPositionFD(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord position estimation based on (F)ixed tail (D)istance.
    """

    def setup(self):
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            [
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:vertical_tail:span",
                "data:geometry:vertical_tail:sweep_25",
                "data:geometry:has_T_tail",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        has_t_tail = inputs["data:geometry:has_T_tail"]

        vt_lp = lp_ht - 0.6 * b_v * math.tan(sweep_25_vt) * has_t_tail

        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = vt_lp

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        has_t_tail = inputs["data:geometry:has_T_tail"]

        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = 1.0
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:vertical_tail:span",
        ] = (
            -0.6 * math.tan(sweep_25_vt) * has_t_tail
        )
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:vertical_tail:sweep_25",
        ] = (
            -0.6 * b_v * has_t_tail / math.cos(sweep_25_vt) ** 2.0
        )
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            "data:geometry:has_T_tail",
        ] = (
            -0.6 * math.tan(sweep_25_vt) * b_v
        )


oad.RegisterSubmodel.active_models[
    SUBMODEL_VT_POSITION_FL
] = "fastga.submodel.geometry.vertical_tail.position.fl.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_VT_POSITION_FL, "fastga.submodel.geometry.vertical_tail.position.fl.legacy"
)
class ComputeVTMacPositionFL(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord position estimation based on (F)ixed fuselage (L)ength (VTP
    distance computed).
    """

    def setup(self):
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:absolute", val=np.nan, units="m"
        )

        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:tip:x", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        x_wing25 = inputs["data:geometry:wing:MAC:at25percent:x"]
        x_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:absolute"]

        tmp = root_chord * 0.25 + b_v * math.tan(sweep_25_vt) - tip_chord * 0.25

        x0_vt = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))

        vt_lp = (x_vt + x0_vt) - x_wing25
        x_tip = b_v * math.tan(sweep_25_vt) + x_wing25 + (vt_lp - x0_vt)

        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt
        outputs["data:geometry:vertical_tail:tip:x"] = x_tip
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = vt_lp
