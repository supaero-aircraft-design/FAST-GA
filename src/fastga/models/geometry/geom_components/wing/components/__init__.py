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
from .compute_wing_l1_l4 import ComputeWingL1AndL4
from .compute_wing_l2_l3 import ComputeWingL2AndL3
from .compute_wing_mac import ComputeWingMAC
from .compute_wing_sweep import ComputeWingSweep
from .compute_wing_toc import ComputeWingToc
from .compute_wing_wet_area import ComputeWingWetArea
from .compute_wing_x import ComputeWingX
from .compute_wing_y import ComputeWingY

# TODO: Determine wether wing drawing shall be included or not
# from .wing_drawing import WingDrawing
