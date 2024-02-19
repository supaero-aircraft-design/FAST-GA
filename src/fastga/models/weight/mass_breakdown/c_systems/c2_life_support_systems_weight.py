"""
Estimation of life support systems weight.
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

from .constants import SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS
] = "fastga.submodel.weight.mass.system.life_support_system.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS,
    "fastga.submodel.weight.mass.system.life_support_system.legacy",
)
class ComputeLifeSupportSystemsWeight(om.ExplicitComponent):
    """
    Weight estimation for life support systems

    This includes only air conditioning / pressurization. Anti-icing is bundled up with the
    air-conditioning weight

    Insulation, internal lighting system, permanent security kits are neglected.
    Seats and installation of crew are already evaluated within d2_passenger_seats_weight

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013` for the air conditioning and de-icing and :cite:`roskampart5:1985`
    for the fixed oxygen weight
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:weight:systems:avionics:mass", val=np.nan, units="lb")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:weight:systems:life_support:insulation:mass", units="lb")
        self.add_output("data:weight:systems:life_support:air_conditioning:mass", units="lb")
        self.add_output("data:weight:systems:life_support:de_icing:mass", units="lb")
        self.add_output("data:weight:systems:life_support:internal_lighting:mass", units="lb")
        self.add_output("data:weight:systems:life_support:seat_installation:mass", units="lb")
        self.add_output("data:weight:systems:life_support:fixed_oxygen:mass", units="lb")
        self.add_output("data:weight:systems:life_support:security_kits:mass", units="lb")

        self.declare_partials(
            "data:weight:systems:life_support:air_conditioning:mass",
            [
                "data:geometry:cabin:seats:passenger:NPAX_max",
                "data:weight:aircraft:MTOW",
                "data:weight:systems:avionics:mass",
                "data:mission:sizing:cs23:characteristic_speed:vd",
            ],
            method="exact",
        )
        self.declare_partials(
            "data:weight:systems:life_support:air_conditioning:mass",
            [
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            method="fd",
            step=1e2,
        )
        self.declare_partials(
            "data:weight:systems:life_support:fixed_oxygen:mass",
            "data:geometry:cabin:seats:passenger:NPAX_max",
            method="exact",
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

        c21 = 0.0

        c22 = (
            0.265 * mtow ** 0.52 * n_occ ** 0.68 * m_iae ** 0.17 * limit_mach ** 0.08
        )  # mass formula in lb

        c23 = 0.0
        c24 = 0.0
        c25 = 0.0

        c26 = 7.0 * n_occ ** 0.702

        c27 = 0.0

        outputs["data:weight:systems:life_support:insulation:mass"] = c21
        outputs["data:weight:systems:life_support:air_conditioning:mass"] = c22
        outputs["data:weight:systems:life_support:de_icing:mass"] = c23
        outputs["data:weight:systems:life_support:internal_lighting:mass"] = c24
        outputs["data:weight:systems:life_support:seat_installation:mass"] = c25
        outputs["data:weight:systems:life_support:fixed_oxygen:mass"] = c26
        outputs["data:weight:systems:life_support:security_kits:mass"] = c27

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

        partials[
            "data:weight:systems:life_support:fixed_oxygen:mass",
            "data:geometry:cabin:seats:passenger:NPAX_max",
        ] = (
            7.0 * 0.702 * n_occ ** -0.298
        )
