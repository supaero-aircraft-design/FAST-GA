"""
Computes the mass of the wing spar web based on the model presented by Raquel ALONSO
in her MAE research project report.
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

import openmdao.api as om
import numpy as np


class ComputeMiscMass(om.ExplicitComponent):
    """Computes the misc mass based on the model developed in FLOPS."""

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_input(
            "settings:wing:structure:F_COMP",
            val=0.0,
            desc="Composite utilisation factor; 1.0 for max composite utilisation, "
            "0.0 for min utilisation",
        )

        self.add_output("data:weight:airframe:wing:misc:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Component that computes the misc mass necessary to react to the given linear force
        vector, according to the methodology developed by NASA's FLOPS.
        """

        wing_area_sq_ft = inputs["data:geometry:wing:area"] * (3.28084 ** 2.0)
        f_comp = inputs["settings:wing:structure:F_COMP"]

        misc_mass = (0.16 * (1.0 - 0.3 * f_comp) * wing_area_sq_ft ** 1.2) * 0.453592

        if inputs["data:geometry:propulsion:engine:count"] > 4:
            misc_mass *= 1.1

        outputs["data:weight:airframe:wing:misc:mass"] = misc_mass
