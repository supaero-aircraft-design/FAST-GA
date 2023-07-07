"""
    Estimation of horizontal tail mean aerodynamic chord properties based on 
    (F)ixed tail (D)istance.
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

from .compute_ht_mac_length import ComputeHTMacLength
from .compute_ht_mac_y import ComputeHTMacY
from .compute_ht_mac25_x_local import ComputeHTMacX25


class ComputeHTMacFD(om.Group):
    def setup(self):

        self.add_subsystem(
            "comp_ht_mac_length_fd",
            ComputeHTMacLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "comp_ht_mac_x25_fd",
            ComputeHTMacX25(),
            promotes=["*"],
        )
        self.add_subsystem(
            "comp_ht_mac_y_fd",
            ComputeHTMacY(),
            promotes=["*"],
        )
