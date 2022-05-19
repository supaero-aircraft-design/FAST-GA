"""
Computes the mass of the secondary structure based on the model presented by Raquel ALONSO
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


class ComputeSecondaryMass(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:airframe:wing:primary_structure:mass", val=np.nan, units="kg")

        self.add_input(
            "settings:wing:structure:secondary_mass_ratio",
            val=0.25,
            desc="Ratio of the mass of the secondary structure and the primary structure (between "
            "0.25 and 0.30 according to literature",
        )

        self.add_output("data:weight:airframe:wing:secondary_structure:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sec_mass_ratio = inputs["settings:wing:structure:secondary_mass_ratio"]
        primary_structure_mass = inputs["data:weight:airframe:wing:primary_structure:mass"]

        total_mass = primary_structure_mass / (1.0 - sec_mass_ratio)
        secondary_structure_mass = total_mass * sec_mass_ratio

        outputs["data:weight:airframe:wing:secondary_structure:mass"] = secondary_structure_mass
