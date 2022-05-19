"""
Computes the mass of the primary structure based on the model presented by Raquel ALONSO
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


class ComputePrimaryMass(om.ExplicitComponent):
    """Computes the primary mass by summing all sub components in the most constraining case."""

    def setup(self):
        self.add_input(
            "data:weight:airframe:wing:web:mass:max_fuel_in_wing", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:airframe:wing:web:mass:min_fuel_in_wing", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:airframe:wing:lower_flange:mass:max_fuel_in_wing", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:airframe:wing:lower_flange:mass:min_fuel_in_wing", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:airframe:wing:upper_flange:mass:max_fuel_in_wing", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:airframe:wing:upper_flange:mass:min_fuel_in_wing", val=np.nan, units="kg"
        )
        self.add_input("data:weight:airframe:wing:skin:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:ribs:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:misc:mass", val=np.nan, units="kg")

        self.add_output("data:weight:airframe:wing:primary_structure:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        primary_mass = (
            max(
                inputs["data:weight:airframe:wing:web:mass:max_fuel_in_wing"],
                inputs["data:weight:airframe:wing:web:mass:min_fuel_in_wing"],
            )
            + max(
                inputs["data:weight:airframe:wing:lower_flange:mass:max_fuel_in_wing"],
                inputs["data:weight:airframe:wing:lower_flange:mass:min_fuel_in_wing"],
            )
            + max(
                inputs["data:weight:airframe:wing:upper_flange:mass:max_fuel_in_wing"],
                inputs["data:weight:airframe:wing:upper_flange:mass:min_fuel_in_wing"],
            )
            + inputs["data:weight:airframe:wing:skin:mass"]
            + inputs["data:weight:airframe:wing:ribs:mass"]
            + inputs["data:weight:airframe:wing:misc:mass"]
        )

        outputs["data:weight:airframe:wing:primary_structure:mass"] = primary_mass
