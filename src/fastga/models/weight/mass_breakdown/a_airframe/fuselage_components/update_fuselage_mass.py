"""
Update the mass of the fuselage based on the computation of the sub-components
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

        shell_mass = inputs["data:weight:airframe:fuselage:shell:mass"]
        cone_mass = inputs["data:weight:airframe:fuselage:cone:mass"]
        windows_mass = inputs["data:weight:airframe:fuselage:windows:mass"]
        insulation_mass = inputs["data:weight:airframe:fuselage:insulation:mass"]
        floor_mass = inputs["data:weight:airframe:fuselage:floor:mass"]
        nlg_hatch_mass = inputs["data:weight:airframe:fuselage:nlg_hatch:mass"]
        doors_mass = inputs["data:weight:airframe:fuselage:doors:mass"]
        wing_fuselage_connection_mass = inputs[
            "data:weight:airframe:fuselage:wing_fuselage_connection:mass"
        ]
        engine_support_mass = inputs["data:weight:airframe:fuselage:engine_support:mass"]
        bulkhead_mass = inputs["data:weight:airframe:fuselage:bulkhead:mass"]
        additional_mass_h = inputs["data:weight:airframe:fuselage:additional_mass:horizontal"]
        additional_mass_v = inputs["data:weight:airframe:fuselage:additional_mass:vertical"]

        fuselage_mass = (
            shell_mass
            + cone_mass
            + windows_mass
            + insulation_mass
            + floor_mass
            + nlg_hatch_mass
            + doors_mass
            + wing_fuselage_connection_mass
            + engine_support_mass
            + bulkhead_mass
            + additional_mass_h
            + additional_mass_v
        )

        outputs["data:weight:airframe:fuselage:mass"] = fuselage_mass
