"""
Python module for wing mean aerodynamic chord calculation, part of the wing geometry.
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

from ..constants import SERVICE_WING_MAC, SUBMODEL_WING_MAC_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_MAC, SUBMODEL_WING_MAC_LEGACY)
class ComputeWingMAC(om.ExplicitComponent):
    """Wing mean aerodynamic chord estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:MAC:length", units="m")
        self.add_output("data:geometry:wing:MAC:leading_edge:x:local", units="m")
        self.add_output("data:geometry:wing:MAC:y", units="m")

        self.declare_partials(
            "data:geometry:wing:MAC:length",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:area",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:geometry:wing:MAC:leading_edge:x:local",
            [
                "data:geometry:wing:tip:leading_edge:x:local",
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:area",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:geometry:wing:MAC:y",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:area",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        l0_wing = (
            3.0 * y2_wing * l2_wing**2.0
            + (y4_wing - y2_wing) * (l2_wing**2.0 + l4_wing**2.0 + l2_wing * l4_wing)
        ) * (2.0 / (3.0 * wing_area))

        x0_wing = (x4_wing * ((y4_wing - y2_wing) * (2.0 * l4_wing + l2_wing))) / (3.0 * wing_area)

        y0_wing = (
            3.0 * y2_wing**2.0 * l2_wing
            + (y4_wing - y2_wing)
            * (l4_wing * (y2_wing + 2.0 * y4_wing) + l2_wing * (y4_wing + 2.0 * y2_wing))
        ) / (3.0 * wing_area)

        outputs["data:geometry:wing:MAC:length"] = l0_wing
        outputs["data:geometry:wing:MAC:leading_edge:x:local"] = x0_wing
        outputs["data:geometry:wing:MAC:y"] = y0_wing

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        partials["data:geometry:wing:MAC:length", "data:geometry:wing:area"] = (
            2.0
            * (
                (y2_wing - y4_wing) * (l2_wing**2.0 + l2_wing * l4_wing + l4_wing**2.0)
                - 3.0 * l2_wing**2.0 * y2_wing
            )
        ) / (3.0 * wing_area**2.0)
        partials["data:geometry:wing:MAC:length", "data:geometry:wing:root:y"] = -(
            2.0 * (-2.0 * l2_wing**2.0 + l2_wing * l4_wing + l4_wing**2.0)
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:length", "data:geometry:wing:tip:y"] = (
            2.0 * (l2_wing**2.0 + l2_wing * l4_wing + l4_wing**2.0)
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:length", "data:geometry:wing:root:chord"] = (
            2.0 * (6.0 * l2_wing * y2_wing - (2.0 * l2_wing + l4_wing) * (y2_wing - y4_wing))
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:length", "data:geometry:wing:tip:chord"] = -(
            2.0 * (l2_wing + 2.0 * l4_wing) * (y2_wing - y4_wing)
        ) / (3.0 * wing_area)

        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:area"] = (
            x4_wing * (l2_wing + 2.0 * l4_wing) * (y2_wing - y4_wing)
        ) / (3.0 * wing_area**2.0)
        partials[
            "data:geometry:wing:MAC:leading_edge:x:local",
            "data:geometry:wing:tip:leading_edge:x:local",
        ] = -((l2_wing + 2.0 * l4_wing) * (y2_wing - y4_wing)) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:root:y"] = -(
            x4_wing * (l2_wing + 2.0 * l4_wing)
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:tip:y"] = (
            x4_wing * (l2_wing + 2.0 * l4_wing)
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:root:chord"] = (
            -(x4_wing * (y2_wing - y4_wing)) / (3.0 * wing_area)
        )
        partials["data:geometry:wing:MAC:leading_edge:x:local", "data:geometry:wing:tip:chord"] = -(
            2.0 * x4_wing * (y2_wing - y4_wing)
        ) / (3.0 * wing_area)

        partials["data:geometry:wing:MAC:y", "data:geometry:wing:area"] = (
            (l2_wing * (2.0 * y2_wing + y4_wing) + l4_wing * (y2_wing + 2.0 * y4_wing))
            * (y2_wing - y4_wing)
            - 3.0 * l2_wing * y2_wing**2.0
        ) / (3.0 * wing_area**2.0)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:root:y"] = -(
            l2_wing * (2.0 * y2_wing + y4_wing)
            - 6.0 * l2_wing * y2_wing
            + l4_wing * (y2_wing + 2.0 * y4_wing)
            + (2.0 * l2_wing + l4_wing) * (y2_wing - y4_wing)
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:tip:y"] = (
            l2_wing * (2.0 * y2_wing + y4_wing)
            + l4_wing * (y2_wing + 2.0 * y4_wing)
            - (l2_wing + 2.0 * l4_wing) * (y2_wing - y4_wing)
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:root:chord"] = -(
            (y2_wing - y4_wing) * (2.0 * y2_wing + y4_wing) - 3.0 * y2_wing**2.0
        ) / (3.0 * wing_area)
        partials["data:geometry:wing:MAC:y", "data:geometry:wing:tip:chord"] = -(
            (y2_wing - y4_wing) * (y2_wing + 2.0 * y4_wing)
        ) / (3.0 * wing_area)
