"""Estimation of wing chords (l1 and l4)."""
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

from ..constants import SERVICE_WING_L1_L4, SUBMODEL_WING_L1_L4_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_L1_L4, SUBMODEL_WING_L1_L4_LEGACY)
class ComputeWingL1AndL4(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing chords (l1 and l4) estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)

        self.add_output("data:geometry:wing:root:virtual_chord", units="m")
        self.add_output("data:geometry:wing:tip:chord", units="m")

        self.declare_partials(of="data:geometry:wing:root:virtual_chord", wrt="*", method="exact")
        self.declare_partials(of="data:geometry:wing:tip:chord", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        l1_wing = wing_area / (2 * y2_wing + (y4_wing - y2_wing) * (1 + taper_ratio))

        l4_wing = l1_wing * taper_ratio

        outputs["data:geometry:wing:root:virtual_chord"] = l1_wing
        outputs["data:geometry:wing:tip:chord"] = l4_wing

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]

        common_denominator = 2 * y2_wing + (y4_wing - y2_wing) * (1 + taper_ratio)

        partials["data:geometry:wing:root:virtual_chord", "data:geometry:wing:area"] = (
            1 / common_denominator
        )
        partials["data:geometry:wing:root:virtual_chord", "data:geometry:wing:root:y"] = (
            -wing_area * (1 - taper_ratio) / common_denominator**2
        )
        partials["data:geometry:wing:root:virtual_chord", "data:geometry:wing:tip:y"] = (
            -wing_area * (1 + taper_ratio) / common_denominator**2
        )
        partials["data:geometry:wing:root:virtual_chord", "data:geometry:wing:taper_ratio"] = (
            -wing_area * (y4_wing - y2_wing) / common_denominator**2
        )

        partials["data:geometry:wing:tip:chord", "data:geometry:wing:area"] = (
            taper_ratio / common_denominator
        )

        partials["data:geometry:wing:tip:chord", "data:geometry:wing:root:y"] = (
            -wing_area * (1 - taper_ratio) * taper_ratio / common_denominator**2
        )
        partials["data:geometry:wing:tip:chord", "data:geometry:wing:tip:y"] = (
            -wing_area * (1 + taper_ratio) * taper_ratio / common_denominator**2
        )
        partials["data:geometry:wing:tip:chord", "data:geometry:wing:taper_ratio"] = (
            wing_area * (y4_wing + y2_wing) / common_denominator**2
        )
