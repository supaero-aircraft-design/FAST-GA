"""
Python module for wing half-span calculation, part of the wing geometry.
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

from ..constants import SERVICE_WING_B50, SUBMODEL_WING_B50_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_B50, SUBMODEL_WING_B50_LEGACY)
class ComputeWingB50(om.ExplicitComponent):
    """Wing B50 estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_50", val=np.nan, units="rad")

        self.add_output("data:geometry:wing:b_50", units="m")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        span = inputs["data:geometry:wing:span"]
        sweep_50 = inputs["data:geometry:wing:sweep_50"]

        b_50 = span / np.cos(sweep_50)

        outputs["data:geometry:wing:b_50"] = b_50

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        span = inputs["data:geometry:wing:span"]
        sweep_50 = inputs["data:geometry:wing:sweep_50"]

        partials["data:geometry:wing:b_50", "data:geometry:wing:span"] = 1.0 / np.cos(sweep_50)
        partials["data:geometry:wing:b_50", "data:geometry:wing:sweep_50"] = (
            span * np.tan(sweep_50) / np.cos(sweep_50)
        )
