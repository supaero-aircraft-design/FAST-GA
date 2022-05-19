"""
Update the mass of the wing based on the computation of the sub-wing_components
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


class UpdateWingMass(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:airframe:wing:primary_structure:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:secondary_structure:mass", val=np.nan, units="kg")

        self.add_output("data:weight:airframe:wing:mass", val=100.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        primary_structure_mass = inputs["data:weight:airframe:wing:primary_structure:mass"]
        secondary_structure_mass = inputs["data:weight:airframe:wing:secondary_structure:mass"]

        wing_mass = primary_structure_mass + secondary_structure_mass

        outputs["data:weight:airframe:wing:mass"] = wing_mass
