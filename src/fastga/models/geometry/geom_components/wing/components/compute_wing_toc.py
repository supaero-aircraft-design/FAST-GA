"""
Python module for wing thickness ratio calculations for different positions, part of the wing
geometry.
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

from ..constants import SERVICE_WING_THICKNESS_RATIO, SUBMODEL_WING_THICKNESS_RATIO_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_THICKNESS_RATIO, SUBMODEL_WING_THICKNESS_RATIO_LEGACY)
class ComputeWingToc(om.ExplicitComponent):
    """Wing ToC estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:geometry:wing:root:thickness_ratio")
        self.add_output("data:geometry:wing:kink:thickness_ratio")
        self.add_output("data:geometry:wing:tip:thickness_ratio")

        self.declare_partials("data:geometry:wing:root:thickness_ratio", "*", val=1.24)
        self.declare_partials("data:geometry:wing:kink:thickness_ratio", "*", val=0.94)
        self.declare_partials("data:geometry:wing:tip:thickness_ratio", "*", val=0.86)

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        el_aero = inputs["data:geometry:wing:thickness_ratio"]

        el_emp = 1.24 * el_aero
        el_break = 0.94 * el_aero
        el_ext = 0.86 * el_aero

        outputs["data:geometry:wing:root:thickness_ratio"] = el_emp
        outputs["data:geometry:wing:kink:thickness_ratio"] = el_break
        outputs["data:geometry:wing:tip:thickness_ratio"] = el_ext
