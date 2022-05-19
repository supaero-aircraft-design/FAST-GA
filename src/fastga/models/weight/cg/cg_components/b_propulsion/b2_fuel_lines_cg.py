"""Estimation of fuel lines center of gravity."""
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


class ComputeFuelLinesCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Fuel lines center of gravity estimation"""

    def setup(self):

        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:fuel_lines:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cg_b1 = inputs["data:weight:propulsion:engine:CG:x"]
        cg_b3 = inputs["data:weight:propulsion:tank:CG:x"]

        cg_b2 = (cg_b1 + cg_b3) / 2.0

        outputs["data:weight:propulsion:fuel_lines:CG:x"] = cg_b2
