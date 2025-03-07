"""
Python module for tank span-wise position computation class(es), part of the advanced MFW
computation method.
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


class ComputeWingTankSpans(om.ExplicitComponent):
    """
    Computes the span-wise position of the start and end of the tank based on geometrical ratio.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output("data:geometry:propulsion:tank:y_beginning", units="m", val=1.0)
        self.add_output("data:geometry:propulsion:tank:y_end", units="m", val=6.0)

        self.declare_partials(
            of="data:geometry:propulsion:tank:y_beginning",
            wrt=[
                "data:geometry:propulsion:tank:y_ratio_tank_beginning",
                "data:geometry:wing:span",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:geometry:propulsion:tank:y_end",
            wrt=[
                "data:geometry:propulsion:tank:y_ratio_tank_end",
                "data:geometry:wing:span",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:geometry:propulsion:tank:y_beginning"] = (
            inputs["data:geometry:propulsion:tank:y_ratio_tank_beginning"]
            * inputs["data:geometry:wing:span"]
            / 2.0
        )
        outputs["data:geometry:propulsion:tank:y_end"] = (
            inputs["data:geometry:propulsion:tank:y_ratio_tank_end"]
            * inputs["data:geometry:wing:span"]
            / 2.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:geometry:propulsion:tank:y_beginning",
            "data:geometry:propulsion:tank:y_ratio_tank_beginning",
        ] = inputs["data:geometry:wing:span"] / 2.0
        partials["data:geometry:propulsion:tank:y_beginning", "data:geometry:wing:span"] = (
            inputs["data:geometry:propulsion:tank:y_ratio_tank_beginning"] / 2.0
        )

        partials[
            "data:geometry:propulsion:tank:y_end",
            "data:geometry:propulsion:tank:y_ratio_tank_end",
        ] = inputs["data:geometry:wing:span"] / 2.0
        partials["data:geometry:propulsion:tank:y_end", "data:geometry:wing:span"] = (
            inputs["data:geometry:propulsion:tank:y_ratio_tank_end"] / 2.0
        )
