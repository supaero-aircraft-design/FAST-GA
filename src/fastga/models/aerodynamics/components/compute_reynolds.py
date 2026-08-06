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
import openmdao.api as om
from stdatm import AtmosphereWithPartials

ATMOSPHERE_0 = AtmosphereWithPartials(altitude=0)


class ComputeUnitReynolds(om.ExplicitComponent):
    """
    Computes the mach number and reynolds number based on inputs and the ISA model.
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        if self.options["low_speed_aero"]:
            self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
            self.add_output("data:aerodynamics:low_speed:mach", units="unitless")
            self.add_output("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
        else:
            self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
            self.add_output("data:aerodynamics:cruise:mach", units="unitless")
            self.add_output("data:aerodynamics:cruise:unit_reynolds", units="m**-1")

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        if self.options["low_speed_aero"]:
            self.declare_partials(
                "data:aerodynamics:low_speed:mach",
                "data:TLAR:v_approach",
                method="exact",
                val=1.0 / ATMOSPHERE_0.speed_of_sound,
            )
        else:
            self.declare_partials(of="*", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["low_speed_aero"]:
            atm = ATMOSPHERE_0
            mach = inputs["data:TLAR:v_approach"] / atm.speed_of_sound
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            atm = AtmosphereWithPartials(altitude, altitude_in_feet=False)
            mach = inputs["data:TLAR:v_cruise"] / atm.speed_of_sound

        atm.mach = mach
        unit_reynolds = atm.unitary_reynolds

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:low_speed:mach"] = mach
            outputs["data:aerodynamics:low_speed:unit_reynolds"] = unit_reynolds
        else:
            outputs["data:aerodynamics:cruise:mach"] = mach
            outputs["data:aerodynamics:cruise:unit_reynolds"] = unit_reynolds

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["low_speed_aero"]:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            atm = AtmosphereWithPartials(altitude, altitude_in_feet=False)
            partials["data:aerodynamics:cruise:mach", "data:TLAR:v_cruise"] = (
                1.0 / atm.speed_of_sound
            )
            partials[
                "data:aerodynamics:cruise:mach", "data:mission:sizing:main_route:cruise:altitude"
            ] = (
                -inputs["data:TLAR:v_cruise"]
                / atm.speed_of_sound**2.0
                * atm.partial_speed_of_sound_altitude
            )

            partials["data:aerodynamics:cruise:unit_reynolds", "data:TLAR:v_cruise"] = (
                1.0 / atm.kinematic_viscosity
            )
            partials[
                "data:aerodynamics:cruise:unit_reynolds",
                "data:mission:sizing:main_route:cruise:altitude",
            ] = (
                -inputs["data:TLAR:v_cruise"]
                / atm.kinematic_viscosity**2.0
                * atm.partial_kinematic_viscosity_altitude
            )
