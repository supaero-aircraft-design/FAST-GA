"""
    Estimation of horizontal tail wet area
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


class ComputeHTWetArea(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Horizontal tail wet area estimation """

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:geometry:horizontal_tail:wet_area", units="m**2")

        self.declare_partials("*", "data:geometry:horizontal_tail:area", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        area = inputs["data:geometry:horizontal_tail:area"]
        tail_type = inputs["data:geometry:has_T_tail"]

        if tail_type == 1.0:
            wet_area_coeff = 1.6 * 1.05  # k_b coef from Gudmunnson p.707
        else:
            wet_area_coeff = 2.0 * 1.05  # k_b coef from Gudmunnson p.707
        wet_area = wet_area_coeff * area

        outputs["data:geometry:horizontal_tail:wet_area"] = wet_area
