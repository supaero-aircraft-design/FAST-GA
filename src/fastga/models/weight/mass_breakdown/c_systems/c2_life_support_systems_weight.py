"""
Estimation of life support systems weight
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
from openmdao.core.explicitcomponent import ExplicitComponent
from fastoad.model_base import Atmosphere


class ComputeLifeSupportSystemsWeight(ExplicitComponent):
    """
    Weight estimation for life support systems

    This includes only air conditioning / pressurization.
    
    Insulation, de-icing, internal lighting system, fixed oxygen, permanent security kits are neglected.
    Seats and installation of crew are already evaluated within d2_passenger_seats_weight

    Based on : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
    Butterworth-Heinemann, 2013. Equation (6-40)
    """

    def setup(self):
        
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:weight:systems:navigation:mass", val=np.nan, units="lb")
        self.add_input("data:TLAR:v_limit", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
       
        self.add_output("data:weight:systems:life_support:insulation:mass", units="lb")
        self.add_output("data:weight:systems:life_support:air_conditioning:mass", units="lb")
        self.add_output("data:weight:systems:life_support:de_icing:mass", units="lb")
        self.add_output("data:weight:systems:life_support:internal_lighting:mass", units="lb")
        self.add_output("data:weight:systems:life_support:seat_installation:mass", units="lb")
        self.add_output("data:weight:systems:life_support:fixed_oxygen:mass", units="lb")
        self.add_output("data:weight:systems:life_support:security_kits:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mtow = inputs["data:weight:aircraft:MTOW"]
        n_pax = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        m_iae = inputs["data:weight:systems:navigation:mass"]
        limit_speed = inputs["data:TLAR:v_limit"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        n_occ = n_pax + 2.
        # Because there are two pilots that needs to be taken into account

        atm = Atmosphere(cruise_alt)
        limit_mach = limit_speed/atm.speed_of_sound  # converted to mach

        c21 = 0.0

        c22 = 0.265*mtow**.52*n_occ**0.68*m_iae**0.17*limit_mach**0.08  # mass formula in lb

        c23 = 0.0
        c24 = 0.0
        c25 = 0.0

        c26 = 7. * n_occ ** 0.702

        c27 = 0.0

        outputs["data:weight:systems:life_support:insulation:mass"] = c21
        outputs["data:weight:systems:life_support:air_conditioning:mass"] = c22
        outputs["data:weight:systems:life_support:de_icing:mass"] = c23
        outputs["data:weight:systems:life_support:internal_lighting:mass"] = c24
        outputs["data:weight:systems:life_support:seat_installation:mass"] = c25
        outputs["data:weight:systems:life_support:fixed_oxygen:mass"] = c26
        outputs["data:weight:systems:life_support:security_kits:mass"] = c27
