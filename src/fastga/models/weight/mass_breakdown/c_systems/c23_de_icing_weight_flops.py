"""
Estimation of anti icing systems weight.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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


class ComputeAntiIcingSystemsWeightFLOPS(om.ExplicitComponent):
    """
    Weight estimation for anti-icing.

    Based on a statistical analysis. See :cite:`wells:2017`.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="ft")
        self.add_input("data:geometry:wing:span", val=np.nan, units="ft")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:propulsion:nacelle:height", val=np.nan, units="ft")
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="ft")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_output("data:weight:systems:life_support:de_icing:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_width = inputs["data:geometry:fuselage:maximum_width"]
        span = inputs["data:geometry:wing:span"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        engine_number = inputs["data:geometry:propulsion:engine:count"]

        nac_diameter = (
            inputs["data:geometry:propulsion:nacelle:height"]
            + inputs["data:geometry:propulsion:nacelle:width"]
        ) / 2.0

        c23 = span / np.cos(sweep_25) + 3.8 * nac_diameter * engine_number + 1.5 * fus_width

        outputs["data:weight:systems:life_support:de_icing:mass"] = c23

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        span = inputs["data:geometry:wing:span"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        nac_height = inputs["data:geometry:propulsion:nacelle:height"]
        nac_width = inputs["data:geometry:propulsion:nacelle:width"]
        engine_number = inputs["data:geometry:propulsion:engine:count"]

        partials[
            "data:weight:systems:life_support:de_icing:mass",
            "data:geometry:wing:span",
        ] = 1.0 / np.cos(sweep_25)
        partials[
            "data:weight:systems:life_support:de_icing:mass",
            "data:geometry:wing:sweep_25",
        ] = (
            span * np.sin(sweep_25) / np.cos(sweep_25) ** 2.0
        )
        partials[
            "data:weight:systems:life_support:de_icing:mass",
            "data:geometry:propulsion:nacelle:height",
        ] = (
            3.8 * engine_number / 2.0
        )
        partials[
            "data:weight:systems:life_support:de_icing:mass",
            "data:geometry:propulsion:nacelle:width",
        ] = (
            3.8 * engine_number / 2.0
        )
        partials[
            "data:weight:systems:life_support:de_icing:mass",
            "data:geometry:propulsion:engine:count",
        ] = (
            3.8 * (nac_height + nac_width) / 2.0
        )
        partials[
            "data:weight:systems:life_support:de_icing:mass",
            "data:geometry:fuselage:maximum_width",
        ] = 1.5
