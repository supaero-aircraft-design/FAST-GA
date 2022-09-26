"""
    Estimation of tank center of gravity
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

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeTankCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Fuel tank center of gravity estimation """

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:tank:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

        cg_tank = (0.35 + 0.65) / 2 * l0_wing
        cg_b3 = fa_length - 0.25 * l0_wing + cg_tank

        outputs["data:weight:propulsion:tank:CG:x"] = cg_b3
