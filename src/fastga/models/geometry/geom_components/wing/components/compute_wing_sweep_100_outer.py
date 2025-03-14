"""
Python module for wing sweep angle calculation at outer 100% of the MAC, part of the wing sweep
angle.
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


class ComputeWingSweep100Outer(om.ExplicitComponent):
    """
    Estimation of outer wing sweep at outer 100% of the MAC, obtained from :cite:`supaero:2014`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:sweep_100_outer", units="rad")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        outputs["data:geometry:wing:sweep_100_outer"] = np.arctan2(
            (x4_wing + l4_wing - l2_wing), (y4_wing - y2_wing)
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        common_denominator = (x4_wing + l4_wing - l2_wing) ** 2.0 + (y4_wing - y2_wing) ** 2.0

        partials[
            "data:geometry:wing:sweep_100_outer", "data:geometry:wing:tip:leading_edge:x:local"
        ] = (y4_wing - y2_wing) / common_denominator
        partials["data:geometry:wing:sweep_100_outer", "data:geometry:wing:root:y"] = (
            x4_wing + l4_wing - l2_wing
        ) / common_denominator
        partials["data:geometry:wing:sweep_100_outer", "data:geometry:wing:tip:y"] = -(
            (x4_wing + l4_wing - l2_wing) / common_denominator
        )
        partials["data:geometry:wing:sweep_100_outer", "data:geometry:wing:root:chord"] = (
            -(y4_wing - y2_wing) / common_denominator
        )

        partials["data:geometry:wing:sweep_100_outer", "data:geometry:wing:tip:chord"] = (
            y4_wing - y2_wing
        ) / common_denominator
