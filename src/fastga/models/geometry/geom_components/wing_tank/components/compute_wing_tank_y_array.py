"""
Python module for span discretization class(es), part of the advanced MFW computation method.
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


class ComputeWingTankYArray(om.ExplicitComponent):
    """
    Computes the span-wise location of the tank cross-section whose are will be computed. Assumes a
    linear interpolation from start to end.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare(
            "number_points_wing_mfw",
            default=50,
            types=int,
            desc="Number of points to use in the computation of the maximum fuel weight using the "
            "advanced model. Reducing that number can improve convergence.",
        )

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        nb_point_wing = self.options["number_points_wing_mfw"]

        self.add_input("data:geometry:propulsion:tank:y_beginning", units="m", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_end", units="m", val=np.nan)

        self.add_output(
            "data:geometry:propulsion:tank:y_array",
            units="m",
            shape=nb_point_wing,
            val=np.linspace(1.0, 6.0, nb_point_wing),
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:y_array",
            wrt=[
                "data:geometry:propulsion:tank:y_beginning",
                "data:geometry:propulsion:tank:y_end",
            ],
            method="exact",
            rows=np.arange(nb_point_wing),
            cols=np.zeros(nb_point_wing),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nb_point_wing = self.options["number_points_wing_mfw"]

        # For reference, np.linspace(a, b, POINTS_NB) is equal to a + (b - a) *
        # np.arange(POINTS_NB) / (POINTS_NB - 1) which is more easily differentiable
        outputs["data:geometry:propulsion:tank:y_array"] = np.linspace(
            inputs["data:geometry:propulsion:tank:y_beginning"][0],
            inputs["data:geometry:propulsion:tank:y_end"][0],
            nb_point_wing,
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        nb_point_wing = self.options["number_points_wing_mfw"]

        partials[
            "data:geometry:propulsion:tank:y_array",
            "data:geometry:propulsion:tank:y_beginning",
        ] = 1.0 - np.arange(nb_point_wing) / (nb_point_wing - 1)
        partials[
            "data:geometry:propulsion:tank:y_array",
            "data:geometry:propulsion:tank:y_end",
        ] = np.arange(nb_point_wing) / (nb_point_wing - 1)
