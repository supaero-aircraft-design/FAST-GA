"""
    Estimation of vertical tail root chord
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

from ..constants import SUBMODEL_VT_ROOT_CHORD


@oad.RegisterSubmodel(
    SUBMODEL_VT_ROOT_CHORD, "fastga.submodel.geometry.vertical_tail.chord.root.legacy"
)
class ComputeVTRootChord(om.ExplicitComponent):
    """
    Estimates vertical tail root chord
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:root:chord", units="m")

        self.declare_partials("data:geometry:vertical_tail:root:chord", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        s_v = float(inputs["data:geometry:vertical_tail:area"])
        taper_v = inputs["data:geometry:vertical_tail:taper_ratio"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        root_chord = s_v * 2 / (1 + taper_v) / b_v

        outputs["data:geometry:vertical_tail:root:chord"] = root_chord
