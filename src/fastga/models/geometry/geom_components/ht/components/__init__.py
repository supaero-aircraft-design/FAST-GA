"""
Estimation of horizontal tail geometry (components).
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

from .compute_ht_distance import ComputeHTDistance
from .compute_ht_mac_length import ComputeHTMacLength
from .compute_ht_mac_y import ComputeHTMacY
from .compute_ht_mac25_x_local import ComputeHTMacX25
from .compute_ht_mac25_x_from_wing import ComputeHTMacX25Wing
from .compute_ht_sweep_0 import ComputeHTSweep0
from .compute_ht_sweep_50 import ComputeHTSweep50
from .compute_ht_sweep_100 import ComputeHTSweep100
from .compute_ht_wet_area import ComputeHTWetArea
from .compute_ht_efficiency import ComputeHTEfficiency
from .compute_ht_volume_coefficient import ComputeHTVolumeCoefficient
from .compute_ht_span import ComputeHTSpan
from .compute_ht_chord_root import ComputeHTRootChord
from .compute_ht_chord_tip import ComputeHTTipChord
