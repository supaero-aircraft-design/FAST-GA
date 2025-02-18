"""
Estimation of geometry of fuselage part A - Cabin (Commercial).
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

class ComputeFuselageNoseLengthFL(om.ExplicitComponent):
    """
    Computes nose length.
    """

    def setup(self):
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:front_length", units="m")

        self.declare_partials(
            of="*",
            wrt=[
                "data:geometry:propulsion:nacelle:length",
                "data:geometry:fuselage:maximum_height",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]

        if prop_layout == 3.0:  # engine located in nose
            lav = nacelle_length
        else:
            lav = 1.7 * h_f

        outputs["data:geometry:fuselage:front_length"] = lav

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]

        if prop_layout == 3.0:
            partials[
                "data:geometry:fuselage:front_length", "data:geometry:propulsion:nacelle:length"
            ] = 1.0
            partials[
                "data:geometry:fuselage:front_length", "data:geometry:fuselage:maximum_height"
            ] = 0.0

        else:
            partials[
                "data:geometry:fuselage:front_length", "data:geometry:propulsion:nacelle:length"
            ] = 0.0
            partials[
                "data:geometry:fuselage:front_length", "data:geometry:fuselage:maximum_height"
            ] = 1.7
