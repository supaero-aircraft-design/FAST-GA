"""
    Estimation of fuselage center of gravity
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
import warnings
from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeFuselageCG(ExplicitComponent):
    """
    Wing center of gravity estimation

    Based on : Roskam, Jan. Airplane Design: Part 5-Component Weight Estimation. DARcorporation, 1985.
    Table 8.1 Center of Gravity Location of Structural Components
    """

    def setup(self):

        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        self.add_output("data:weight:airframe:fuselage:CG:x", units="m")

        self.declare_partials("data:weight:airframe:fuselage:CG:x", "data:geometry:fuselage:length", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        prop_layout = inputs["data:geometry:propulsion:layout"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        
        # Fuselage gravity center
        if prop_layout == 1.0:
            x_cg_a2 = 0.39 * fus_length
        elif prop_layout == 3.0:  # nose mount
            # x_cg_a2 = lav + 0.35 * (fus_length - lav)
            x_cg_a2 = lav + 0.45 * (fus_length - lav)
        else:
            x_cg_a2 = lav + 0.47 * (fus_length - lav)
        
        outputs["data:weight:airframe:fuselage:CG:x"] = x_cg_a2
