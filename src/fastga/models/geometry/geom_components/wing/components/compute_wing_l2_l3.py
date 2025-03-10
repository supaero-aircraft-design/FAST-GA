"""
Python module for wing chords of calculations (l2 and l3), part of the wing geometry.
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

from ..constants import SERVICE_WING_L2_L3, SUBMODEL_WING_L2_L3_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_L2_L3, SUBMODEL_WING_L2_L3_LEGACY)
class ComputeWingL2AndL3(om.ExplicitComponent):
    """Wing chords (l2 and l3) estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)

        self.add_output("data:geometry:wing:root:chord", units="m")
        self.add_output("data:geometry:wing:kink:chord", units="m")

        self.declare_partials(of="data:geometry:wing:root:chord", wrt="*", method="exact")
        self.declare_partials(of="data:geometry:wing:kink:chord", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        l2_wing = wing_area / (2.0 * y2_wing + (y4_wing - y2_wing) * (1.0 + taper_ratio))

        l3_wing = l2_wing

        outputs["data:geometry:wing:root:chord"] = l2_wing
        outputs["data:geometry:wing:kink:chord"] = l3_wing

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        common_denominator = 2.0 * y2_wing + (y4_wing - y2_wing) * (1.0 + taper_ratio)

        partials["data:geometry:wing:root:chord", "data:geometry:wing:area"] = (
            1.0 / common_denominator
        )
        partials["data:geometry:wing:root:chord", "data:geometry:wing:root:y"] = (
            -wing_area * (1.0 - taper_ratio) / common_denominator**2.0
        )
        partials["data:geometry:wing:root:chord", "data:geometry:wing:tip:y"] = (
            -wing_area * (1.0 + taper_ratio) / common_denominator**2.0
        )
        partials["data:geometry:wing:root:chord", "data:geometry:wing:taper_ratio"] = (
            -wing_area * (y4_wing - y2_wing) / common_denominator**2.0
        )

        partials["data:geometry:wing:kink:chord", "data:geometry:wing:area"] = (
            1.0 / common_denominator
        )
        partials["data:geometry:wing:kink:chord", "data:geometry:wing:root:y"] = (
            -wing_area * (1.0 - taper_ratio) / common_denominator**2.0
        )
        partials["data:geometry:wing:kink:chord", "data:geometry:wing:tip:y"] = (
            -wing_area * (1.0 + taper_ratio) / common_denominator**2.0
        )
        partials["data:geometry:wing:kink:chord", "data:geometry:wing:taper_ratio"] = (
            -wing_area * (y4_wing - y2_wing) / common_denominator**2.0
        )
