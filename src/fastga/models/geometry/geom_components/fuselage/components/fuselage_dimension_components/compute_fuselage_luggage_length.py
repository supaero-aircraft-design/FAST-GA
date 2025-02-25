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


class ComputeFuselageLuggageLength(om.ExplicitComponent):
    """
    Computes luggage length.

    80% of internal radius section can be filled with luggage.
    """

    def setup(self):
        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")

        self.add_output("data:geometry:fuselage:luggage_length", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]

        if 2.0 * w_pilot_seats != seats_p_row * w_pass_seats + w_aisle:
            w_cabin = max(2.0 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)
        else:
            w_cabin = (2.0 * w_pilot_seats + seats_p_row * w_pass_seats + w_aisle) / 2

        r_i = w_cabin / 2

        luggage_density = 161.0  # In kg/m3
        l_lug = (luggage_mass_max / luggage_density) / (0.8 * np.pi * r_i**2.0)

        outputs["data:geometry:fuselage:luggage_length"] = l_lug

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]

        w_cabin = max(2.0 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)
        r_i = w_cabin / 2

        luggage_density = 161.0  # In kg/m3

        partials[
            "data:geometry:fuselage:luggage_length", "data:geometry:cabin:luggage:mass_max"
        ] = 1.0 / (luggage_density * (0.8 * np.pi * r_i**2.0))

        volume_constant = (luggage_mass_max / luggage_density) / (0.8 * np.pi)

        if (2.0 * w_pilot_seats) > (seats_p_row * w_pass_seats + w_aisle):
            partials[
                "data:geometry:fuselage:luggage_length", "data:geometry:cabin:seats:pilot:width"
            ] = -2.0 * volume_constant / w_pilot_seats**3.0

            partials[
                "data:geometry:fuselage:luggage_length",
                "data:geometry:cabin:seats:passenger:count_by_row",
            ] = 0.0

            partials[
                "data:geometry:fuselage:luggage_length", "data:geometry:cabin:seats:passenger:width"
            ] = 0.0

            partials["data:geometry:fuselage:luggage_length", "data:geometry:cabin:aisle_width"] = (
                0.0
            )

        elif (2.0 * w_pilot_seats) < (seats_p_row * w_pass_seats + w_aisle):

            partials[
                "data:geometry:fuselage:luggage_length", "data:geometry:cabin:seats:pilot:width"
            ] = 0.0

            partials[
                "data:geometry:fuselage:luggage_length",
                "data:geometry:cabin:seats:passenger:count_by_row",
            ] = -volume_constant * w_pass_seats / r_i ** 3.0

            partials[
                "data:geometry:fuselage:luggage_length", "data:geometry:cabin:seats:passenger:width"
            ] = -volume_constant * seats_p_row / r_i ** 3.0

            partials["data:geometry:fuselage:luggage_length", "data:geometry:cabin:aisle_width"] = (
                    -volume_constant / r_i ** 3.0
            )

        elif (2.0 * w_pilot_seats) == (seats_p_row * w_pass_seats + w_aisle):
            r_i = 0.25*(2.0 * w_pilot_seats + seats_p_row * w_pass_seats + w_aisle)

            partials[
                "data:geometry:fuselage:luggage_length", "data:geometry:cabin:luggage:mass_max"
            ] = 1.0 / (luggage_density * (0.8 * np.pi * r_i ** 2.0))

            partials[
                "data:geometry:fuselage:luggage_length", "data:geometry:cabin:seats:pilot:width"
            ] = -2*volume_constant / r_i**3.0

            partials[
                "data:geometry:fuselage:luggage_length",
                "data:geometry:cabin:seats:passenger:count_by_row",
            ] = -volume_constant * w_pass_seats / r_i ** 3.0

            partials[
                "data:geometry:fuselage:luggage_length", "data:geometry:cabin:seats:passenger:width"
            ] = -volume_constant * seats_p_row / r_i ** 3.0
            partials["data:geometry:fuselage:luggage_length", "data:geometry:cabin:aisle_width"] = (
                    -volume_constant / r_i ** 3.0
            )
