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

# pylint: disable=unused-import
# flake8: noqa

from .compute_ht_chords import ComputeHTChord
from .compute_ht_distance import ComputeHTDistance
from .compute_ht_efficiency import ComputeHTEfficiency
from .compute_ht_mac import ComputeHTMacFD, ComputeHTMacFL
from .compute_ht_sweep import ComputeHTSweep
from .compute_ht_volume_coefficient import ComputeHTVolumeCoefficient
from .compute_ht_wet_area import ComputeHTWetArea
