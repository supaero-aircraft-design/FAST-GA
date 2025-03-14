"""
Python module for passenger number calculation with fixed fuselage length, part of the fuselage
dimension.
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


class ComputeFuselageNPAX(om.ExplicitComponent):
    """
    Computes number of pax cabin.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)

        self.add_output("data:geometry:cabin:NPAX")

        self.declare_partials(of="*", wrt="*", method="fd")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]

        # noinspection PyBroadException
        npax = np.ceil(npax_max / seats_p_row) * seats_p_row

        outputs["data:geometry:cabin:NPAX"] = npax
