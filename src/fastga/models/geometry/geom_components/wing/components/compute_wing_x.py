"""
Python module for leading edge position calculation, part of the wing geometry.
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

from ..constants import SERVICE_WING_X_LOCAL, SUBMODEL_WING_X_LOCAL_LEGACY


@oad.RegisterSubmodel(SERVICE_WING_X_LOCAL, SUBMODEL_WING_X_LOCAL_LEGACY)
class ComputeWingX(om.ExplicitComponent):
    """Wing Xs estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_output("data:geometry:wing:kink:leading_edge:x:local", units="m")
        self.add_output("data:geometry:wing:tip:leading_edge:x:local", units="m")

        self.declare_partials("*", "data:geometry:wing:root:virtual_chord", val=0.25)
        self.declare_partials(
            "data:geometry:wing:kink:leading_edge:x:local",
            "data:geometry:wing:kink:chord",
            val=-0.25,
        )
        self.declare_partials(
            "data:geometry:wing:tip:leading_edge:x:local",
            "data:geometry:wing:tip:chord",
            val=-0.25,
        )

        self.declare_partials(
            "data:geometry:wing:kink:leading_edge:x:local",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:kink:y",
                "data:geometry:wing:sweep_25",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:geometry:wing:tip:leading_edge:x:local",
            [
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:sweep_25",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l3_wing = inputs["data:geometry:wing:kink:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]

        x3_wing = 0.25 * l1_wing + (y3_wing - y2_wing) * np.tan(sweep_25) - 0.25 * l3_wing
        x4_wing = 0.25 * l1_wing + (y4_wing - y2_wing) * np.tan(sweep_25) - 0.25 * l4_wing

        outputs["data:geometry:wing:kink:leading_edge:x:local"] = x3_wing
        outputs["data:geometry:wing:tip:leading_edge:x:local"] = x4_wing

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]

        partials[
            "data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:root:y"
        ] = -np.tan(sweep_25)
        partials["data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:kink:y"] = (
            np.tan(sweep_25)
        )
        partials["data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:sweep_25"] = (
            y3_wing - y2_wing
        ) / np.cos(sweep_25) ** 2.0

        partials[
            "data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:root:y"
        ] = -np.tan(sweep_25)
        partials["data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:tip:y"] = (
            np.tan(sweep_25)
        )
        partials["data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:sweep_25"] = (
            y4_wing - y2_wing
        ) / np.cos(sweep_25) ** 2.0
