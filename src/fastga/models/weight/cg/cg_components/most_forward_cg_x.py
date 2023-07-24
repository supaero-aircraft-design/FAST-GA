"""
    Estimation of maximum center of gravity ratio.
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

from .constants import SUBMODEL_AIRCRAFT_FORWARD_CG_X


@oad.RegisterSubmodel(
    SUBMODEL_AIRCRAFT_FORWARD_CG_X, "fastga.submodel.weight.cg.aircraft.most_forward.x.legacy"
)
class ComputeForwardCGX(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Extreme (most forward) center of gravity ratio estimation in meter"""

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:MAC_position", val=np.nan)

        self.add_output("data:weight:aircraft:CG:fwd:x", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        mac_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        cg_min_fwd_mac = inputs["data:weight:aircraft:CG:fwd:MAC_position"]

        outputs["data:weight:aircraft:CG:fwd:x"] = (
            mac_position - 0.25 * l0_wing + cg_min_fwd_mac * l0_wing
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_min_fwd_mac = inputs["data:weight:aircraft:CG:fwd:MAC_position"]

        partials["data:weight:aircraft:CG:fwd:x", "data:geometry:wing:MAC:length"] = (
            -0.25 + cg_min_fwd_mac
        )
        partials["data:weight:aircraft:CG:fwd:x", "data:geometry:wing:MAC:at25percent:x"] = 1.0
        partials[
            "data:weight:aircraft:CG:fwd:x", "data:weight:aircraft:CG:fwd:MAC_position"
        ] = l0_wing
