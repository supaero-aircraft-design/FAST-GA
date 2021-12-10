"""
Update the mass of the fuselage based on the computation of the sub-components
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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


class UpdateFuselageMass(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:airframe:fuselage:shell:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:cone:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:windows:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:insulation:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:floor:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:nlg_hatch:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:doors:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:airframe:fuselage:wing_fuselage_connection:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:airframe:fuselage:engine_support:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:bulkhead:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:airframe:fuselage:additional_mass:horizontal", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:airframe:fuselage:additional_mass:vertical", val=np.nan, units="kg"
        )

        self.add_output("data:weight:airframe:fuselage:mass", val=100.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuselage_mass = (
            inputs["data:weight:airframe:fuselage:shell:mass"]
            + inputs["data:weight:airframe:fuselage:cone:mass"]
            + inputs["data:weight:airframe:fuselage:windows:mass"]
            + inputs["data:weight:airframe:fuselage:insulation:mass"]
            + inputs["data:weight:airframe:fuselage:floor:mass"]
            + inputs["data:weight:airframe:fuselage:nlg_hatch:mass"]
            + inputs["data:weight:airframe:fuselage:doors:mass"]
            + inputs["data:weight:airframe:fuselage:wing_fuselage_connection:mass"]
            + inputs["data:weight:airframe:fuselage:engine_support:mass"]
            + inputs["data:weight:airframe:fuselage:bulkhead:mass"]
            + inputs["data:weight:airframe:fuselage:additional_mass:horizontal"]
            + inputs["data:weight:airframe:fuselage:additional_mass:vertical"]
        )

        outputs["data:weight:airframe:fuselage:mass"] = fuselage_mass
