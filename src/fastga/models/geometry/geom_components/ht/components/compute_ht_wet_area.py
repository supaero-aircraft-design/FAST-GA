"""
Python module for horizontal tail wet area calculation, part of the horizontal tail geometry.
"""

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

from ..constants import SERVICE_HT_WET_AREA, SUBMODEL_HT_WET_AREA_LEGACY


@oad.RegisterSubmodel(SERVICE_HT_WET_AREA, SUBMODEL_HT_WET_AREA_LEGACY)
class ComputeHTWetArea(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail wet area estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:geometry:horizontal_tail:wet_area", units="m**2")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs["data:geometry:horizontal_tail:area"]
        tail_type = inputs["data:geometry:has_T_tail"]

        wet_area = (2.0 - 0.4 * tail_type) * 1.05 * area

        outputs["data:geometry:horizontal_tail:wet_area"] = wet_area

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        area = inputs["data:geometry:horizontal_tail:area"]
        tail_type = inputs["data:geometry:has_T_tail"]

        partials["data:geometry:horizontal_tail:wet_area", "data:geometry:has_T_tail"] = (
            -0.4 * 1.05 * area
        )
        partials["data:geometry:horizontal_tail:wet_area", "data:geometry:horizontal_tail:area"] = (
            2.0 - 0.4 * tail_type
        ) * 1.05
