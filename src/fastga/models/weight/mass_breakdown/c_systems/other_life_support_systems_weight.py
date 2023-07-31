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

import openmdao.api as om


class ComputeOtherLifeSupportSystemsWeight(om.ExplicitComponent):
    """
    Weight estimation for life support systems other than anti-icing, air condition, and oxygen.

    Insulation, internal lighting system, permanent security kits are neglected.
    Seats and installation of crew are already evaluated within d2_passenger_seats_weight
    """

    def setup(self):

        self.add_output("data:weight:systems:life_support:insulation:mass", val=0.0, units="lb")
        self.add_output(
            "data:weight:systems:life_support:internal_lighting:mass", val=0.0, units="lb"
        )
        self.add_output(
            "data:weight:systems:life_support:seat_installation:mass", val=0.0, units="lb"
        )
        self.add_output("data:weight:systems:life_support:security_kits:mass", val=0.0, units="lb")
