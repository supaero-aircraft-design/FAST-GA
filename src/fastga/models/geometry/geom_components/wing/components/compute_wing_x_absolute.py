"""
Python module for distance calculation between the nose and leading edge at different section,
part of the wing geometry.
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

from ..constants import SERVICE_WING_X_ABSOLUTE, SUBMODEL_WING_X_ABSOLUTE_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_X_ABSOLUTE, SUBMODEL_WING_X_ABSOLUTE_LEGACY)
class ComputeWingXAbsolute(om.ExplicitComponent):
    """
    Wing absolute Xs estimation, distance from the nose to the leading edge at different
    section.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", units="m", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", units="m", val=np.nan)
        self.add_input("data:geometry:wing:MAC:at25percent:x", units="m", val=np.nan)
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", units="m", val=np.nan)

        self.add_output("data:geometry:wing:tip:leading_edge:x:absolute", units="m")
        self.add_output("data:geometry:wing:MAC:leading_edge:x:absolute", units="m")

        self.declare_partials("*", "data:geometry:wing:MAC:at25percent:x", val=1.0)
        self.declare_partials("*", "data:geometry:wing:MAC:length", val=-0.25)
        self.declare_partials(
            of="data:geometry:wing:tip:leading_edge:x:absolute",
            wrt="data:geometry:wing:MAC:leading_edge:x:local",
            val=-1.0,
        )
        self.declare_partials(
            of="data:geometry:wing:tip:leading_edge:x:absolute",
            wrt="data:geometry:wing:tip:leading_edge:x:local",
            val=1.0,
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_local_mac = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        x_local_tip = inputs["data:geometry:wing:tip:leading_edge:x:local"]

        x_abs_mac = fa_length - 0.25 * l0_wing
        x_abs_tip = x_abs_mac - x_local_mac + x_local_tip

        outputs["data:geometry:wing:MAC:leading_edge:x:absolute"] = x_abs_mac
        outputs["data:geometry:wing:tip:leading_edge:x:absolute"] = x_abs_tip
