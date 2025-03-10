"""
Python module for span calculations of different wing sections, part of the wing geometry.
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

from ..constants import SERVICE_WING_SPAN, SUBMODEL_WING_SPAN_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_SPAN, SUBMODEL_WING_SPAN_LEGACY)
class ComputeWingY(om.ExplicitComponent):
    """Wing Ys estimation, , obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:kink:span_ratio", val=0.5)

        self.add_output("data:geometry:wing:span", units="m")
        self.add_output("data:geometry:wing:root:y", units="m")
        self.add_output("data:geometry:wing:kink:y", units="m")
        self.add_output("data:geometry:wing:tip:y", units="m")

        self.declare_partials(
            of="data:geometry:wing:span",
            wrt=["data:geometry:wing:area", "data:geometry:wing:aspect_ratio"],
            method="exact",
        )
        self.declare_partials(
            of="data:geometry:wing:root:y",
            wrt="data:geometry:fuselage:maximum_width",
            method="exact",
            val=0.5,
        )
        self.declare_partials(
            of="data:geometry:wing:kink:y",
            wrt=[
                "data:geometry:wing:area",
                "data:geometry:wing:aspect_ratio",
                "data:geometry:wing:kink:span_ratio",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:geometry:wing:tip:y",
            wrt=[
                "data:geometry:wing:area",
                "data:geometry:wing:aspect_ratio",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_break = inputs["data:geometry:wing:kink:span_ratio"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        span = np.sqrt(lambda_wing * wing_area)

        # Wing geometry
        y4_wing = span / 2.0
        y2_wing = width_max / 2.0
        y3_wing = y4_wing * wing_break

        outputs["data:geometry:wing:span"] = span
        outputs["data:geometry:wing:root:y"] = y2_wing
        outputs["data:geometry:wing:kink:y"] = y3_wing
        outputs["data:geometry:wing:tip:y"] = y4_wing

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_break = inputs["data:geometry:wing:kink:span_ratio"]

        partials["data:geometry:wing:span", "data:geometry:wing:aspect_ratio"] = 0.5 * np.sqrt(
            wing_area / lambda_wing
        )
        partials["data:geometry:wing:span", "data:geometry:wing:area"] = 0.5 * np.sqrt(
            lambda_wing / wing_area
        )

        partials["data:geometry:wing:tip:y", "data:geometry:wing:aspect_ratio"] = 0.25 * np.sqrt(
            wing_area / lambda_wing
        )
        partials["data:geometry:wing:tip:y", "data:geometry:wing:area"] = 0.25 * np.sqrt(
            lambda_wing / wing_area
        )

        partials["data:geometry:wing:kink:y", "data:geometry:wing:aspect_ratio"] = (
            0.5 * np.sqrt(wing_area / lambda_wing) * wing_break
        )
        partials["data:geometry:wing:kink:y", "data:geometry:wing:area"] = (
            0.5 * np.sqrt(lambda_wing / wing_area) * wing_break
        )
        partials["data:geometry:wing:kink:y", "data:geometry:wing:kink:span_ratio"] = 0.5 * np.sqrt(
            lambda_wing * wing_area
        )
