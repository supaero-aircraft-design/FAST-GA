"""
Estimation of fixed oxygen systems weight.
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


class ComputeFixedOxygenSystemsWeight(om.ExplicitComponent):
    """
    Weight estimation for fixed oxygen system.

    Based on a statistical analysis. See :cite:`roskampart5:1985`.
    """

    def setup(self):

        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)

        self.add_output("data:weight:systems:life_support:fixed_oxygen:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        n_pax = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]

        n_occ = n_pax + 2.0
        # Because there are two pilots that needs to be taken into account

        c26 = 7.0 * n_occ ** 0.702

        outputs["data:weight:systems:life_support:fixed_oxygen:mass"] = c26

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n_pax = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]

        n_occ = n_pax + 2.0

        partials[
            "data:weight:systems:life_support:fixed_oxygen:mass",
            "data:geometry:cabin:seats:passenger:NPAX_max",
        ] = (
            7.0 * 0.702 * n_occ ** -0.298
        )
