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

from stdatm import Atmosphere


class ComputeAirConditioningSystemsWeightFLOPS(om.ExplicitComponent):
    """
    Weight estimation for air conditioning / pressurization.

    Based on a statistical analysis. See :cite:`wells:2017`.
    """

    def setup(self):

        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:weight:systems:avionics:mass", val=np.nan, units="lb")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")

        self.add_output("data:weight:systems:life_support:air_conditioning:mass", units="lb")

        self.declare_partials(
            of="*",
            wrt=[
                "data:geometry:cabin:seats:passenger:NPAX_max",
                "data:weight:systems:avionics:mass",
                "data:mission:sizing:cs23:characteristic_speed:vd",
                "data:geometry:fuselage:maximum_height",
                "data:geometry:fuselage:maximum_width",
                "data:geometry:fuselage:length",
            ],
            method="exact",
        )
        self.declare_partials(
            of="*", wrt="data:mission:sizing:main_route:cruise:altitude", method="fd", step=1e2
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_iae = inputs["data:weight:systems:avionics:mass"]
        limit_speed = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        fus_height = inputs["data:geometry:fuselage:maximum_width"]
        fus_width = inputs["data:geometry:fuselage:maximum_height"]

        fus_planform = fus_width * inputs["data:geometry:fuselage:length"]

        n_occ = inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        # Because there are two pilots that needs to be taken into account

        atm = Atmosphere(cruise_alt, altitude_in_feet=True)
        limit_mach = limit_speed / atm.speed_of_sound  # converted to mach

        c22 = (
            3.2 * (fus_planform * fus_height) ** 0.6 + 9 * n_occ ** 0.83
        ) * limit_mach + 0.075 * m_iae
        # mass formula in lb

        outputs["data:weight:systems:life_support:air_conditioning:mass"] = c22

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        fus_width = inputs["data:geometry:fuselage:maximum_width"]
        fus_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]

        fus_planform = fus_width * fus_length

        n_occ = inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        # Because there are two pilots that needs to be taken into account

        speed_of_sound = Atmosphere(
            inputs["data:mission:sizing:main_route:cruise:altitude"], altitude_in_feet=True
        ).speed_of_sound
        limit_mach = (
            inputs["data:mission:sizing:cs23:characteristic_speed:vd"] / speed_of_sound
        )  # converted to mach

        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:geometry:cabin:seats:passenger:NPAX_max",
        ] = (
            limit_mach * 9 * 0.83 * n_occ ** -0.17
        )
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:weight:systems:avionics:mass",
        ] = 0.075
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:mission:sizing:cs23:characteristic_speed:vd",
        ] = (3.2 * (fus_planform * fus_height) ** 0.6 + 9 * n_occ ** 0.83) / speed_of_sound
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:geometry:fuselage:maximum_height",
        ] = (
            3.2 * 0.6 * fus_planform / (fus_planform * fus_height) ** 0.4 * limit_mach
        )
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:geometry:fuselage:maximum_width",
        ] = (
            3.2 * 0.6 / (fus_height * fus_planform) ** 0.4 * fus_height * fus_length * limit_mach
        )
        partials[
            "data:weight:systems:life_support:air_conditioning:mass",
            "data:geometry:fuselage:length",
        ] = (
            3.2 * 0.6 * (fus_height * fus_width) ** 0.6 * fus_length ** -0.4 * limit_mach
        )
