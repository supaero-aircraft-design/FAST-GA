"""
Estimation of wing geometry (components).
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

from .compute_wing_b50 import ComputeWingB50
from .compute_wing_l1 import ComputeWingL1
from .compute_wing_l2 import ComputeWingL2
from .compute_wing_l3 import ComputeWingL3
from .compute_wing_l4 import ComputeWingL4
from .compute_wing_mac_length import ComputeWingMacLength
from .compute_wing_mac_x import ComputeWingMacX
from .compute_wing_mac_y import ComputeWingMacY
from .compute_wing_sweep_0 import ComputeWingSweep0
from .compute_wing_sweep_50 import ComputeWingSweep50
from .compute_wing_sweep_100_inner import ComputeWingSweep100Inner
from .compute_wing_sweep_100_outer import ComputeWingSweep100Outer
from .compute_wing_toc_root import ComputeWingTocRoot
from .compute_wing_toc_kink import ComputeWingTocKink
from .compute_wing_toc_tip import ComputeWingTocTip
from .compute_wing_wet_area import ComputeWingWetArea
from .compute_wing_outer_area import ComputeWingOuterArea
from .compute_wing_x_kink import ComputeWingXKink
from .compute_wing_x_tip import ComputeWingXTip
from .compute_wing_y_root import ComputeWingYRoot
from .compute_wing_y_kink import ComputeWingYKink
from .compute_wing_y_tip import ComputeWingYTip
from .compute_wing_span import ComputeWingSpan
from .compute_wing_z_root import ComputeWingZRoot
from .compute_wing_z_tip import ComputeWingZTip
from .compute_wing_x_absolute_mac import ComputeWingXAbsoluteMac
from .compute_wing_x_absolute_tip import ComputeWingXAbsoluteTip
