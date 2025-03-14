"""
Python module for nacelle Y - position calculation, part of the nacelle position.
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

import warnings
import numpy as np
import openmdao.api as om


class ComputeNacelleYPosition(om.ExplicitComponent):
    """
    Estimates y position of the nacelle.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")

        self.add_output(
            "data:geometry:propulsion:nacelle:y",
            units="m",
            shape_by_conn=True,
            copy_shape="data:geometry:propulsion:engine:y_ratio",
        )

        self.declare_partials(of="*", wrt="data:geometry:propulsion:engine:layout", method="fd")

        self.declare_partials(
            of="*",
            wrt=[
                "data:geometry:wing:span",
                "data:geometry:propulsion:engine:y_ratio",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:propulsion:nacelle:width",
            ],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nac_width = inputs["data:geometry:propulsion:nacelle:width"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])
        b_f = inputs["data:geometry:fuselage:maximum_width"]

        if prop_layout == 1.0:
            y_nacelle_array = y_ratio * span / 2.0
        elif prop_layout == 2.0:
            y_nacelle_array = b_f / 2.0 + 0.8 * nac_width
        elif prop_layout == 3.0:
            y_nacelle_array = 0.0
        else:
            y_nacelle_array = 0.0
            warnings.warn(
                f"Propulsion layout {prop_layout} not implemented in model, "
                f"replaced by layout 3!",
                category=UserWarning,
            )

        outputs["data:geometry:propulsion:nacelle:y"] = y_nacelle_array

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        span = inputs["data:geometry:wing:span"]
        y_ratio = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])

        if prop_layout == 1.0:
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:propulsion:nacelle:width"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:y", "data:geometry:wing:span"] = (
                y_ratio / 2.0
            )
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:propulsion:engine:y_ratio"
            ] = span / 2.0
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:fuselage:maximum_width"
            ] = 0.0
        elif prop_layout == 2.0:
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:propulsion:nacelle:width"
            ] = 0.8
            partials["data:geometry:propulsion:nacelle:y", "data:geometry:wing:span"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:propulsion:engine:y_ratio"
            ] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:fuselage:maximum_width"
            ] = 0.5
        else:
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:propulsion:nacelle:width"
            ] = 0.0
            partials["data:geometry:propulsion:nacelle:y", "data:geometry:wing:span"] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:propulsion:engine:y_ratio"
            ] = 0.0
            partials[
                "data:geometry:propulsion:nacelle:y", "data:geometry:fuselage:maximum_width"
            ] = 0.0
