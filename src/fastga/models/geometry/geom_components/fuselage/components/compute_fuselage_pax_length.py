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
import fastoad.api as oad

from ..constants import SUBMODEL_FUSELAGE_PAX_LENGTH


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_PAX_LENGTH, "fastga.submodel.geometry.fuselage.dimensions.pax_length.legacy"
)
class ComputeFuselagePAXLength(om.ExplicitComponent):
    """
    Computes Length of pax cabin.
    """

    def setup(self):

        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:NPAX", val=np.nan)

        self.add_output("data:geometry:fuselage:PAX_length", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l_pilot_seats = inputs["data:geometry:cabin:seats:pilot:length"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        npax = inputs["data:geometry:cabin:NPAX"]

        n_rows = npax / float(seats_p_row)
        l_pax = l_pilot_seats + n_rows * l_pass_seats

        outputs["data:geometry:fuselage:PAX_length"] = l_pax

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        npax = inputs["data:geometry:cabin:NPAX"]

        partials[
            "data:geometry:fuselage:PAX_length", "data:geometry:cabin:seats:pilot:length"
        ] = 1.0
        partials[
            "data:geometry:fuselage:PAX_length", "data:geometry:cabin:seats:passenger:length"
        ] = npax / float(seats_p_row)
        partials[
            "data:geometry:fuselage:PAX_length", "data:geometry:cabin:seats:passenger:count_by_row"
        ] = (-npax / (float(seats_p_row)) ** 2 * l_pass_seats)
        partials[
            "data:geometry:fuselage:PAX_length", "data:geometry:cabin:NPAX"
        ] = l_pass_seats / float(seats_p_row)
