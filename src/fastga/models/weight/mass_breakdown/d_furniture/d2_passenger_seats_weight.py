"""
Python module for passenger seats weight calculation, part of the furniture mass computation.
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om

from .constants import SERVICE_SEATS_MASS, SUBMODEL_SEATS_MASS_LEGACY


@oad.RegisterSubmodel(SERVICE_SEATS_MASS, SUBMODEL_SEATS_MASS_LEGACY)
class ComputePassengerSeatsWeight(om.ExplicitComponent):
    """
    Weight estimation for passenger seats

    Based on a statistical analysis. See :cite:`roskampart5:1985` Cessna method.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")

        self.add_output("data:weight:furniture:passenger_seats:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_occ = (
            inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        )  # includes 2 pilots seats
        mtow = inputs["data:weight:aircraft:MTOW"]

        d2 = 0.412 * n_occ**1.145 * mtow**0.489  # mass formula in lb

        outputs["data:weight:furniture:passenger_seats:mass"] = d2

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n_occ = (
            inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        )  # includes 2 pilots seats
        mtow = inputs["data:weight:aircraft:MTOW"]

        partials["data:weight:furniture:passenger_seats:mass", "data:weight:aircraft:MTOW"] = (
            0.412 * 0.489 * n_occ**1.145
        ) / mtow**0.511

        partials[
            "data:weight:furniture:passenger_seats:mass",
            "data:geometry:cabin:seats:passenger:NPAX_max",
        ] = 0.47174 * n_occ**0.145 * mtow**0.489
