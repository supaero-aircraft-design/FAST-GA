"""Computes the mass of the wing fuselage connection, adapted from TASOPT by Lucas REMOND."""
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


class ComputeWingFuselageConnection(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:mission:landing:cs23:sizing_factor:ultimate_aircraft", val=6.0)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input(
            "data:geometry:cabin:pressurized",
            val=0.0,
            desc="Cabin pressurization; 0.0 for no pressurization, 1.0 for pressurization",
        )

        self.add_output("data:weight:airframe:fuselage:wing_fuselage_connection:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        n_ult_flight = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        n_ult_landing = inputs["data:mission:landing:cs23:sizing_factor:ultimate_aircraft"]
        n_ult = max(n_ult_flight, n_ult_landing)

        mtow = inputs["data:weight:aircraft:MTOW"]

        pressurized = inputs["data:geometry:cabin:pressurized"]

        if pressurized:
            mass_wing_fuselage_connection = 20.4 + 0.907e-3 * n_ult * mtow
        else:
            mass_wing_fuselage_connection = 0.4e-3 * (n_ult * mtow) ** 1.185

        outputs[
            "data:weight:airframe:fuselage:wing_fuselage_connection:mass"
        ] = mass_wing_fuselage_connection
