"""Estimation of tank center of gravity."""
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

from ..constants import SUBMODEL_TANK_CG


@oad.RegisterSubmodel(SUBMODEL_TANK_CG, "fastga.submodel.weight.cg.propulsion.tank.legacy")
class ComputeTankCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Fuel tank center of gravity estimation"""

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_input(
            "settings:weight:propulsion:tank:CG:from_wingMAC25",
            val=0.25,
            desc="distance between the tank CG and 25 percent of wing MAC as a ratio of the wing "
            "MAC",
        )

        self.add_output("data:weight:propulsion:tank:CG:x", units="m")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

        distance_from_25_mac_ratio = inputs["settings:weight:propulsion:tank:CG:from_wingMAC25"]

        cg_b3 = fa_length + distance_from_25_mac_ratio * l0_wing

        outputs["data:weight:propulsion:tank:CG:x"] = cg_b3

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        distance_from_25_mac_ratio = inputs["settings:weight:propulsion:tank:CG:from_wingMAC25"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]

        partials["data:weight:propulsion:tank:CG:x", "data:geometry:wing:MAC:at25percent:x"] = 1.0
        partials[
            "data:weight:propulsion:tank:CG:x", "data:geometry:wing:MAC:length"
        ] = distance_from_25_mac_ratio
        partials[
            "data:weight:propulsion:tank:CG:x", "settings:weight:propulsion:tank:CG:from_wingMAC25"
        ] = l0_wing
