"""
Estimation of geometry of fuselage part A - Cabin (Commercial). 
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

from ..constants import SUBMODEL_FUSELAGE_MAX_HEIGHT


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_MAX_HEIGHT, "fastga.submodel.geometry.fuselage.dimensions.max_height.legacy"
)
class ComputeFuselageMaxHeight(om.ExplicitComponent):
    """
    Computes maximum cabin height.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:maximum_height", units="m")

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]

        # 0.14m is the distance between both lobe centers of the fuselage
        h_f = b_f + 0.14

        outputs["data:geometry:fuselage:maximum_height"] = h_f
