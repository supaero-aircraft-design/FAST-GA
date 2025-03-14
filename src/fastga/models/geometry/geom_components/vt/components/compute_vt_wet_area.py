"""
Python module for vertical tail wet area calculation, part of the vertical tail geometry.
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

from ..constants import SERVICE_VT_WET_AREA, SUBMODEL_VT_WET_AREA_LEGACY


@oad.RegisterSubmodel(SERVICE_VT_WET_AREA, SUBMODEL_VT_WET_AREA_LEGACY)
class ComputeVTWetArea(om.ExplicitComponent):
    """Vertical tail wet area estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")

        self.add_output("data:geometry:vertical_tail:wet_area", units="m**2")

        self.declare_partials("*", "*", val=2.1)

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs["data:geometry:vertical_tail:area"]

        wet_area = 2.1 * area

        outputs["data:geometry:vertical_tail:wet_area"] = wet_area
