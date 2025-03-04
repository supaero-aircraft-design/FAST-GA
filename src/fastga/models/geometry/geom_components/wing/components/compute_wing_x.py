"""Estimation of wing Xs."""
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

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ..constants import SUBMODEL_WING_X_LOCAL


@oad.RegisterSubmodel(SUBMODEL_WING_X_LOCAL, "fastga.submodel.geometry.wing.x_local.legacy")
class ComputeWingX(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Wing Xs estimation."""

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

        self.declare_partials(
            "data:geometry:wing:kink:leading_edge:x:local",
            [
                "data:geometry:wing:root:virtual_chord",
                "data:geometry:wing:root:y",
                "data:geometry:wing:kink:y",
                "data:geometry:wing:kink:chord",
                "data:geometry:wing:sweep_25",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:geometry:wing:tip:leading_edge:x:local",
            [
                "data:geometry:wing:root:virtual_chord",
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:sweep_25",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        y2_wing = inputs["data:geometry:wing:root:y"]
        y3_wing = inputs["data:geometry:wing:kink:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l3_wing = inputs["data:geometry:wing:kink:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]

        x3_wing = 1.0 / 4.0 * l1_wing + (y3_wing - y2_wing) * np.tan(sweep_25) - 1.0 / 4.0 * l3_wing
        x4_wing = 1.0 / 4.0 * l1_wing + (y4_wing - y2_wing) * np.tan(sweep_25) - 1.0 / 4.0 * l4_wing

        outputs["data:geometry:wing:kink:leading_edge:x:local"] = x3_wing
        outputs["data:geometry:wing:tip:leading_edge:x:local"] = x4_wing

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
        partials[
            "data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:root:virtual_chord"
        ] = 0.25
        partials[
            "data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:kink:chord"
        ] = -0.25
        partials["data:geometry:wing:kink:leading_edge:x:local", "data:geometry:wing:sweep_25"] = -(
            np.tan(sweep_25) ** 2.0 + 1.0
        ) * (y2_wing - y3_wing)

        partials[
            "data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:root:y"
        ] = -np.tan(sweep_25)
        partials["data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:tip:y"] = (
            np.tan(sweep_25)
        )
        partials[
            "data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:root:virtual_chord"
        ] = 0.25
        partials[
            "data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:tip:chord"
        ] = -0.25
        partials["data:geometry:wing:tip:leading_edge:x:local", "data:geometry:wing:sweep_25"] = -(
            np.tan(sweep_25) ** 2.0 + 1.0
        ) * (y2_wing - y4_wing)
