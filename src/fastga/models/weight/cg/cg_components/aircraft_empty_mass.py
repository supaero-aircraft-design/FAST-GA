"""
    Estimation of aircraft empty mass.
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

from ..cg_components.constants import SUBMODEL_AIRCRAFT_EMPTY_MASS


@oad.RegisterSubmodel(
    SUBMODEL_AIRCRAFT_EMPTY_MASS, "fastga.submodel.weight.cg.aircraft_empty.mass.legacy"
)
class ComputeEmptyMass(om.ExplicitComponent):
    """
    Computes aircraft empty mass.
    """

    def initialize(self):

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

        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:mass", units="kg")

        for mass_name in self.options["mass_names"]:
            self.declare_partials("data:weight:aircraft_empty:mass", mass_name, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]

        outputs["data:weight:aircraft_empty:mass"] = np.sum(masses)
