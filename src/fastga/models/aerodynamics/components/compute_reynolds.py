"""
Computes Mach number and unitary Reynolds.
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
from openmdao.core.explicitcomponent import ExplicitComponent
from stdatm import Atmosphere


class ComputeUnitReynolds(ExplicitComponent):
    """
    Computes the mach number and reynolds number based on inputs and the ISA model.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
            self.add_output("data:aerodynamics:low_speed:mach")
            self.add_output("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
        else:
            self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
            self.add_output("data:aerodynamics:cruise:mach")
            self.add_output("data:aerodynamics:cruise:unit_reynolds", units="m**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["low_speed_aero"]:
            altitude = 0.0
            mach = inputs["data:TLAR:v_approach"] / Atmosphere(altitude).speed_of_sound
        else:
            altitude = float(inputs["data:mission:sizing:main_route:cruise:altitude"])
            mach = (
                inputs["data:TLAR:v_cruise"]
                / Atmosphere(altitude, altitude_in_feet=False).speed_of_sound
            )

        atm = Atmosphere(altitude, altitude_in_feet=False)
        atm.mach = mach
        unit_reynolds = atm.unitary_reynolds

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:low_speed:mach"] = mach
            outputs["data:aerodynamics:low_speed:unit_reynolds"] = unit_reynolds
        else:
            outputs["data:aerodynamics:cruise:mach"] = mach
            outputs["data:aerodynamics:cruise:unit_reynolds"] = unit_reynolds
