"""Estimation of vertical tail tip position."""

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

from ..constants import SUBMODEL_VT_POSITION_TIP_X


@oad.RegisterSubmodel(
    SUBMODEL_VT_POSITION_TIP_X, "fastga.submodel.geometry.vertical_tail.position.tip_x.legacy"
)
class ComputeVTXTip(om.ExplicitComponent):
    """
    Computes x coordinate of the vertical tail's tip.
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )

        self.add_output("data:geometry:vertical_tail:tip:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        x_wing25 = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:local"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]

        x_tip = b_v * np.tan(sweep_25_vt) + x_wing25 + (vt_lp - x0_vt)

        outputs["data:geometry:vertical_tail:tip:x"] = x_tip
