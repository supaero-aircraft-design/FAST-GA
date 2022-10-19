"""
    Estimation of center of gravity ratio with aft.
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

import numpy as np
import openmdao.api as om

from fastoad.module_management.service_registry import RegisterSubmodel

from fastga.models.weight.cg.cg_components.constants import SUBMODEL_AIRCRAFT_X_CG

@RegisterSubmodel(SUBMODEL_AIRCRAFT_X_CG, "fastga.submodel.weight.cg.aircraft_empty.x.hybrid.fuel_cell")
class ComputeCGFuelCell(om.ExplicitComponent):
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
                "data:weight:propulsion:CG:x",
                "data:weight:systems:power:electric_systems:CG:x",
                "data:weight:systems:power:hydraulic_systems:CG:x",
                "data:weight:systems:life_support:air_conditioning:CG:x",
                "data:weight:systems:avionics:CG:x",
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
                "data:weight:propulsion:mass",
                "data:weight:systems:power:electric_systems:mass",
                "data:weight:systems:power:hydraulic_systems:mass",
                "data:weight:systems:life_support:air_conditioning:mass",
                "data:weight:systems:avionics:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
        )

    def setup(self):
        for cg_name in self.options["cg_names"]:
            self.add_input(cg_name, val=np.nan, units="m")
        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:mass", units="kg")
        self.add_output("data:weight:aircraft_empty:CG:x", units="m")

        self.declare_partials("data:weight:aircraft_empty:mass", "*", method="fd")
        self.declare_partials("data:weight:aircraft_empty:CG:x", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cgs = [inputs[cg_name][0] for cg_name in self.options["cg_names"]]
        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]

        weight_moment = np.dot(cgs, masses)
        outputs["data:weight:aircraft_empty:mass"] = np.sum(masses)
        x_cg_empty_aircraft = weight_moment / outputs["data:weight:aircraft_empty:mass"]
        outputs["data:weight:aircraft_empty:CG:x"] = x_cg_empty_aircraft
