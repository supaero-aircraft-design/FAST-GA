"""
    Estimation of aircraft empty center of gravity ratio.
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


class ComputeCGRatio(om.ExplicitComponent):
    """
    Computes X-position of center of gravity as ratio of mean aerodynamic chord for empty aircraft.
    """

    def setup(self):

        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:empty:CG:MAC_position")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x_cg_all = inputs["data:weight:aircraft_empty:CG:x"]
        wing_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        mac = inputs["data:geometry:wing:MAC:length"]

        outputs["data:weight:aircraft:empty:CG:MAC_position"] = (
            x_cg_all - wing_position + 0.25 * mac
        ) / mac

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        x_cg_all = inputs["data:weight:aircraft_empty:CG:x"]
        wing_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        mac = inputs["data:geometry:wing:MAC:length"]

        partials[
            "data:weight:aircraft:empty:CG:MAC_position", "data:weight:aircraft_empty:CG:x"
        ] = (1.0 / mac)
        partials[
            "data:weight:aircraft:empty:CG:MAC_position", "data:geometry:wing:MAC:at25percent:x"
        ] = (-1.0 / mac)
        partials["data:weight:aircraft:empty:CG:MAC_position", "data:geometry:wing:MAC:length"] = (
            -(x_cg_all - wing_position) / mac ** 2.0
        )
