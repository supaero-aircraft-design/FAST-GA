"""
Python module for tank capacity computation class(es), part of the advanced MFW computation
method.
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

from scipy.integrate import trapezoid


class ComputeWingTanksCapacity(om.ExplicitComponent):
    """Compute the capacity of the two wing tanks inside the aircraft wings."""

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
        self.add_input(
            "data:geometry:propulsion:tank:cross_section_array",
            shape=nb_point_wing,
            units="m**2",
            val=np.nan,
        )
        self.add_input(
            "data:geometry:propulsion:tank:y_array",
            shape=nb_point_wing,
            units="m",
            val=np.nan,
        )

        self.add_output(
            "data:geometry:propulsion:tank:capacity",
            units="m**3",
            val=1.0,
            desc="Capacity of both tanks on the aircraft",
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:capacity",
            wrt="*",
            method="exact",
            rows=np.zeros(nb_point_wing),
            cols=np.arange(nb_point_wing),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        y_array = inputs["data:geometry:propulsion:tank:y_array"]
        cross_section_array = inputs["data:geometry:propulsion:tank:cross_section_array"]

        # trapz should be equivalent to sum(
        #   (
        #       (cross_section_array[:-1] + cross_section_array[1:])
        #       / 2.0
        #       * (y_array[1:] - y_array[:-1])
        #   )
        #
        outputs["data:geometry:propulsion:tank:capacity"] = 2.0 * trapezoid(
            cross_section_array, y_array
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y_array = inputs["data:geometry:propulsion:tank:y_array"]
        cross_section_array = inputs["data:geometry:propulsion:tank:cross_section_array"]

        first_order_delta_y = y_array[1:] - y_array[:-1]
        second_order_delta_y = y_array[2:] - y_array[:-2]

        first_order_sigma_cs = cross_section_array[:-1] + cross_section_array[1:]
        second_order_delta_cs = cross_section_array[:-2] - cross_section_array[2:]

        partials_cross_section = np.concatenate(
            (
                np.array([first_order_delta_y[0]]),
                second_order_delta_y,
                np.array([first_order_delta_y[-1]]),
            )
        )
        partials_y = np.concatenate(
            (
                np.array([-first_order_sigma_cs[0]]),
                second_order_delta_cs,
                np.array([first_order_sigma_cs[-1]]),
            )
        )

        partials[
            "data:geometry:propulsion:tank:capacity",
            "data:geometry:propulsion:tank:y_array",
        ] = partials_y
        partials[
            "data:geometry:propulsion:tank:capacity",
            "data:geometry:propulsion:tank:cross_section_array",
        ] = partials_cross_section
