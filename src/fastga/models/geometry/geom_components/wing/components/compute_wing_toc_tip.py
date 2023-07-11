"""Estimation of wing tip ToC."""
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

from ..constants import SUBMODEL_WING_THICKNESS_RATIO_TIP


# TODO: computes relative thickness and generates profiles --> decompose
@oad.RegisterSubmodel(
    SUBMODEL_WING_THICKNESS_RATIO_TIP, "fastga.submodel.geometry.wing.thickness_ratio.tip"
)
class ComputeWingTocTip(om.ExplicitComponent):
    """Wing tip ToC estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:geometry:wing:tip:thickness_ratio")

        self.declare_partials("*", "*", val=0.86)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        el_aero = inputs["data:geometry:wing:thickness_ratio"]

        el_ext = 0.86 * el_aero

        outputs["data:geometry:wing:tip:thickness_ratio"] = el_ext
