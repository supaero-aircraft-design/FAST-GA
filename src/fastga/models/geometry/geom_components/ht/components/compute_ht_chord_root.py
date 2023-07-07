"""
    Estimation of horizontal tail chords and span
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


class ComputeHTRootChord(om.ExplicitComponent):
    """Root chord estimation of horizontal tail"""

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:root:chord", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        s_h = inputs["data:geometry:horizontal_tail:area"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        root_chord = s_h * 2 / (1 + taper_ht) / b_h

        outputs["data:geometry:horizontal_tail:root:chord"] = root_chord
