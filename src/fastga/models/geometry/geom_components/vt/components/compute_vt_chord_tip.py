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

from ..constants import SUBMODEL_VT_TIP_CHORD


@oad.RegisterSubmodel(
    SUBMODEL_VT_TIP_CHORD, "fastga.submodel.geometry.vertical_tail.chord.tip.legacy"
)
class ComputeVTTipChord(om.ExplicitComponent):
    """
    Estimates vertical tail tip chord
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:tip:chord", units="m")

        self.declare_partials("data:geometry:vertical_tail:tip:chord", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        taper_v = inputs["data:geometry:vertical_tail:taper_ratio"]
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]

        tip_chord = root_chord * taper_v

        outputs["data:geometry:vertical_tail:tip:chord"] = tip_chord
