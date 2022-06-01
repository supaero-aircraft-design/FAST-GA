"""Estimation of flight control center of gravity."""
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
from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ..constants import SUBMODEL_FLIGHT_CONTROLS_CG


@oad.RegisterSubmodel(
    SUBMODEL_FLIGHT_CONTROLS_CG, "fastga.submodel.weight.cg.airframe.flight_controls.legacy"
)
class ComputeFlightControlCG(ExplicitComponent):
    """
    Control surfaces center of gravity estimation.

    Based on the position of the aerodynamic center of all lifting surfaces. Not taken at the exact
    position of the control surfaces as flight controls weight includes cockpit controls and
    pulleys/cables.
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )

        self.add_output("data:weight:airframe:flight_controls:CG:x", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]

        x_cg_a4 = 0.5 * fa_length + 0.25 * (fa_length + lp_ht) + 0.25 * (fa_length + lp_vt)

        outputs["data:weight:airframe:flight_controls:CG:x"] = x_cg_a4

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials[
            "data:weight:airframe:flight_controls:CG:x", "data:geometry:wing:MAC:at25percent:x"
        ] = 1
        partials[
            "data:weight:airframe:flight_controls:CG:x",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = 0.25
        partials[
            "data:weight:airframe:flight_controls:CG:x",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = 0.25
