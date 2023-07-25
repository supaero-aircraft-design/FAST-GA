"""
    Estimation of aircraft empty center of gravity.
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

from ..cg_components.constants import SUBMODEL_AIRCRAFT_X_CG


@oad.RegisterSubmodel(SUBMODEL_AIRCRAFT_X_CG, "fastga.submodel.weight.cg.aircraft_empty.x.legacy")
class ComputeCG(om.ExplicitComponent):
    """
    Computes aircraft empty center of gravity.
    """

    def initialize(self):

        self.options.declare(
            "cg_names",
            default=[
                "data:weight:airframe:wing:CG:x",
                "data:weight:airframe:fuselage:CG:x",
                "data:weight:airframe:horizontal_tail:CG:x",
                "data:weight:airframe:vertical_tail:CG:x",
                "data:weight:airframe:flight_controls:CG:x",
                "data:weight:airframe:landing_gear:main:CG:x",
                "data:weight:airframe:landing_gear:front:CG:x",
                "data:weight:propulsion:engine:CG:x",
                "data:weight:propulsion:fuel_lines:CG:x",
                "data:weight:systems:power:electric_systems:CG:x",
                "data:weight:systems:power:hydraulic_systems:CG:x",
                "data:weight:systems:life_support:air_conditioning:CG:x",
                "data:weight:systems:avionics:CG:x",
                "data:weight:systems:recording:CG:x",
                "data:weight:furniture:passenger_seats:CG:x",
            ],
        )
        self.options.declare(
            "mass_names",
            [
                "data:weight:airframe:wing:mass",
                "data:weight:airframe:fuselage:mass",
                "data:weight:airframe:horizontal_tail:mass",
                "data:weight:airframe:vertical_tail:mass",
                "data:weight:airframe:flight_controls:mass",
                "data:weight:airframe:landing_gear:main:mass",
                "data:weight:airframe:landing_gear:front:mass",
                "data:weight:propulsion:engine:mass",
                "data:weight:propulsion:fuel_lines:mass",
                "data:weight:systems:power:electric_systems:mass",
                "data:weight:systems:power:hydraulic_systems:mass",
                "data:weight:systems:life_support:air_conditioning:mass",
                "data:weight:systems:avionics:mass",
                "data:weight:systems:recording:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
        )

    def setup(self):

        for cg_name in self.options["cg_names"]:
            self.add_input(cg_name, val=np.nan, units="m")
        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")
        self.add_input("data:weight:aircraft_empty:mass", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:CG:x", units="m")

        self.declare_partials("data:weight:aircraft_empty:CG:x", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cgs = [inputs[cg_name][0] for cg_name in self.options["cg_names"]]
        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]
        total_mass = inputs["data:weight:aircraft_empty:mass"]

        weight_moment = np.dot(cgs, masses)
        x_cg_empty_aircraft = weight_moment / total_mass

        outputs["data:weight:aircraft_empty:CG:x"] = x_cg_empty_aircraft

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        total_mass = inputs["data:weight:aircraft_empty:mass"]

        weight_moment = 0.0
        for cg_name, mass_name in zip(self.options["cg_names"], self.options["mass_names"]):
            partials["data:weight:aircraft_empty:CG:x", cg_name] = inputs[mass_name] / total_mass
            partials["data:weight:aircraft_empty:CG:x", mass_name] = inputs[cg_name] / total_mass
            weight_moment += inputs[cg_name] * inputs[mass_name]

        partials["data:weight:aircraft_empty:CG:x", "data:weight:aircraft_empty:mass"] = (
            -weight_moment / total_mass ** 2.0
        )
