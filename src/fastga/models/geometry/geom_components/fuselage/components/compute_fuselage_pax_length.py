"""
Python module for fuselage passenger compartment length calculation, part of the fuselage dimension.
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


class ComputeFuselagePAXLength(om.ExplicitComponent):
    """
    Computes Length of passenger area of the cabin.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:NPAX", val=np.nan)

        self.add_output("data:geometry:fuselage:PAX_length", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")
        self.declare_partials(
            "data:geometry:fuselage:PAX_length", "data:geometry:cabin:seats:pilot:length", val=1.0
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        l_pilot_seats = inputs["data:geometry:cabin:seats:pilot:length"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        npax = inputs["data:geometry:cabin:NPAX"]

        n_rows = npax / seats_p_row
        l_pax = l_pilot_seats + n_rows * l_pass_seats

        outputs["data:geometry:fuselage:PAX_length"] = l_pax

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        npax = inputs["data:geometry:cabin:NPAX"]

        partials[
            "data:geometry:fuselage:PAX_length", "data:geometry:cabin:seats:passenger:length"
        ] = npax / seats_p_row
        partials[
            "data:geometry:fuselage:PAX_length", "data:geometry:cabin:seats:passenger:count_by_row"
        ] = -npax / seats_p_row**2.0 * l_pass_seats
        partials["data:geometry:fuselage:PAX_length", "data:geometry:cabin:NPAX"] = (
            l_pass_seats / seats_p_row
        )
