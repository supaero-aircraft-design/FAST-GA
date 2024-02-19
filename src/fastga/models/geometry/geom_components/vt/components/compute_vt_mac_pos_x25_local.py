"""Estimation of vertical tail mean aerodynamic chord position (local)."""

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

from ..constants import SUBMODEL_VT_POSITION_X25_LOCAL


@oad.RegisterSubmodel(
    SUBMODEL_VT_POSITION_X25_LOCAL,
    "fastga.submodel.geometry.vertical_tail.position.fl.x25_local.legacy",
)
class ComputeVTMacX25Local(om.ExplicitComponent):
    """
    Computes x coordinate (local) at 25% MAC of the vertical tail.
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        tmp = root_chord * 0.25 + b_v * np.tan(sweep_25_vt) - tip_chord * 0.25
        x0_vt = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        tmp = 3 * root_chord + 3 * tip_chord

        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:root:chord",
        ] = (
            (root_chord + 2 * tip_chord) / (4 * tmp)
            + (root_chord / 4 - tip_chord / 4 + b_v * np.tan(sweep_25_vt)) / tmp
            - (
                3
                * (root_chord + 2 * tip_chord)
                * (root_chord / 4 - tip_chord / 4 + b_v * np.tan(sweep_25_vt))
            )
            / tmp ** 2
        )
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:tip:chord",
        ] = (
            (2 * (root_chord / 4 - tip_chord / 4 + b_v * np.tan(sweep_25_vt))) / tmp
            - (root_chord + 2 * tip_chord) / (4 * tmp)
            - (
                3
                * (root_chord + 2 * tip_chord)
                * (root_chord / 4 - tip_chord / 4 + b_v * np.tan(sweep_25_vt))
            )
            / tmp ** 2
        )
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:sweep_25",
        ] = (b_v * (root_chord + 2 * tip_chord) * (np.tan(sweep_25_vt) ** 2 + 1)) / tmp
        partials[
            "data:geometry:vertical_tail:MAC:at25percent:x:local",
            "data:geometry:vertical_tail:span",
        ] = (np.tan(sweep_25_vt) * (root_chord + 2 * tip_chord)) / tmp
