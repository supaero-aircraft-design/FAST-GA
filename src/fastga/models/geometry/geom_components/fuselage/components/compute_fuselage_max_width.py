"""
Python module for fuselage maximum width calculation, part of the fuselage dimension.
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


class ComputeFuselageMaxWidth(om.ExplicitComponent):
    """
    Computes maximum cabin width.

    Cabin width considered is for side by side seats, and it is computed based on
    cylindrical fuselage.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:maximum_width", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]

        w_cabin = max(2.0 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)

        b_f = 1.06 * w_cabin

        outputs["data:geometry:fuselage:maximum_width"] = b_f

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]

        if (2.0 * w_pilot_seats) > (seats_p_row * w_pass_seats + w_aisle):
            partials[
                "data:geometry:fuselage:maximum_width", "data:geometry:cabin:seats:pilot:width"
            ] = 2.12
            partials[
                "data:geometry:fuselage:maximum_width", "data:geometry:cabin:seats:passenger:width"
            ] = 0.0
            partials[
                "data:geometry:fuselage:maximum_width",
                "data:geometry:cabin:seats:passenger:count_by_row",
            ] = 0.0
            partials["data:geometry:fuselage:maximum_width", "data:geometry:cabin:aisle_width"] = (
                0.0
            )

        else:
            partials[
                "data:geometry:fuselage:maximum_width", "data:geometry:cabin:seats:pilot:width"
            ] = 0.0
            partials[
                "data:geometry:fuselage:maximum_width", "data:geometry:cabin:seats:passenger:width"
            ] = 1.06 * seats_p_row
            partials[
                "data:geometry:fuselage:maximum_width",
                "data:geometry:cabin:seats:passenger:count_by_row",
            ] = 1.06 * w_pass_seats
            partials["data:geometry:fuselage:maximum_width", "data:geometry:cabin:aisle_width"] = (
                1.06
            )
