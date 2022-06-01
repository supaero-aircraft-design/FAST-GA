"""Estimation of vertical tail wet area."""
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

from ..constants import SUBMODEL_VT_WET_AREA


@oad.RegisterSubmodel(
    SUBMODEL_VT_WET_AREA, "fastga.submodel.geometry.vertical_tail.wet_area.legacy"
)
class ComputeVTWetArea(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail wet area estimation"""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")

        self.add_output("data:geometry:vertical_tail:wet_area", units="m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs["data:geometry:vertical_tail:area"]

        wet_area = 2.1 * area

        outputs["data:geometry:vertical_tail:wet_area"] = wet_area
