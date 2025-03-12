"""
Python module for the calculation of horizontal tail distance from 25% wing MAC, part of the
horizontal tail geometry.
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

from ..constants import SERVICE_HT_DISTANCE, SUBMODEL_HT_DISTANCE_LEGACY


@oad.RegisterSubmodel(SERVICE_HT_DISTANCE, SUBMODEL_HT_DISTANCE_LEGACY)
class ComputeHTDistance(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail distance estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:geometry:horizontal_tail:z:from_wingMAC25", units="m")

        self.declare_partials(
            "data:geometry:horizontal_tail:z:from_wingMAC25",
            ["data:geometry:vertical_tail:span", "data:geometry:has_T_tail"],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tail_type = inputs["data:geometry:has_T_tail"]
        span = inputs["data:geometry:vertical_tail:span"]

        outputs["data:geometry:horizontal_tail:z:from_wingMAC25"] = span * tail_type

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        tail_type = inputs["data:geometry:has_T_tail"]
        span = inputs["data:geometry:vertical_tail:span"]

        partials["data:geometry:horizontal_tail:z:from_wingMAC25", "data:geometry:has_T_tail"] = (
            span
        )
        partials[
            "data:geometry:horizontal_tail:z:from_wingMAC25", "data:geometry:vertical_tail:span"
        ] = tail_type
