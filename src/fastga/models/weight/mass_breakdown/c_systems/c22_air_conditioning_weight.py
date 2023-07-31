"""
Estimation of air conditioning system weight.
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

from stdatm import Atmosphere


class ComputeAirConditioningSystemsWeight(om.ExplicitComponent):
    """
    Weight estimation air conditioning / pressurization.

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013` for the air conditioning and de-icing.
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:weight:systems:avionics:mass", val=np.nan, units="lb")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:weight:systems:life_support:air_conditioning:mass", units="lb")

        self.declare_partials(
            of="data:weight:systems:life_support:air_conditioning:mass",
            wrt=[
                "data:geometry:cabin:seats:passenger:NPAX_max",
                "data:weight:aircraft:MTOW",
                "data:weight:systems:avionics:mass",
                "data:mission:sizing:cs23:characteristic_speed:vd",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:weight:systems:life_support:air_conditioning:mass",
            wrt=[
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            method="fd",
            step=1e2,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        n_pax = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        m_iae = inputs["data:weight:systems:avionics:mass"]
        limit_speed = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        n_occ = n_pax + 2.0
        # Because there are two pilots that needs to be taken into account

        atm = Atmosphere(cruise_alt, altitude_in_feet=True)
        limit_mach = limit_speed / atm.speed_of_sound  # converted to mach

        c22 = (
            0.265 * mtow ** 0.52 * n_occ ** 0.68 * m_iae ** 0.17 * limit_mach ** 0.08
        )  # mass formula in lb

        outputs["data:weight:systems:life_support:air_conditioning:mass"] = c22

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        n_pax = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        m_iae = inputs["data:weight:systems:avionics:mass"]
        limit_speed = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        speed_of_sound = Atmosphere(cruise_alt, altitude_in_feet=True).speed_of_sound

        n_occ = n_pax + 2.0
        limit_mach = limit_speed / speed_of_sound

        partials[
            "data:weight:systems:life_support:air_conditioning:mass", "data:weight:aircraft:MTOW"
        ] = (0.52 * 0.265 * mtow ** -0.48 * n_occ ** 0.68 * m_iae ** 0.17 * limit_mach ** 0.08)
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:geometry:cabin:seats:passenger:NPAX_max",
        ] = (
            0.68 * 0.265 * mtow ** 0.52 * n_occ ** -0.32 * m_iae ** 0.17 * limit_mach ** 0.08
        )
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:weight:systems:avionics:mass",
        ] = (
            0.17 * 0.265 * mtow ** 0.52 * n_occ ** 0.68 * m_iae ** -0.83 * limit_mach ** 0.08
        )
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:mission:sizing:cs23:characteristic_speed:vd",
        ] = (
            0.08 * 0.265 * mtow ** 0.52 * n_occ ** 0.68 * m_iae ** 0.17 * limit_mach ** -0.92
        ) / speed_of_sound
