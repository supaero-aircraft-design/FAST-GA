"""
    Estimation of horizontal tail efficiency
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

from openmdao.core.explicitcomponent import ExplicitComponent
import fastoad.api as oad

from ..constants import SUBMODEL_HT_WET_EFFICIENCY


@oad.RegisterSubmodel(
    SUBMODEL_HT_WET_EFFICIENCY, "fastga.submodel.geometry.horizontal_tail.efficiency.legacy"
)
class ComputeHTEfficiency(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail dynamic pressure reduction due to geometric positioning"""

    def setup(self):
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:aerodynamics:horizontal_tail:efficiency")

        self.declare_partials(
            of="data:aerodynamics:horizontal_tail:efficiency",
            wrt="data:geometry:has_T_tail",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:aerodynamics:horizontal_tail:efficiency"] = (
            0.9 + 0.1 * inputs["data:geometry:has_T_tail"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["data:aerodynamics:horizontal_tail:efficiency", "data:geometry:has_T_tail"] = 0.1
