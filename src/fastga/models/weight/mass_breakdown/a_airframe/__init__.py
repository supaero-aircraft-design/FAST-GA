"""Package containing the subcomponents necessary for the airframe mass estimation."""
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

# flake8: noqa

from .a1_wing_weight import ComputeWingWeight
from .a1_wing_weight_analytical import ComputeWingMassAnalytical
from .a2_fuselage_weight import ComputeFuselageWeight
from .a2_fuselage_weight_raymer import ComputeFuselageWeightRaymer
from .a2_fuselage_weight_roskam import ComputeFuselageWeightRoskam
from .a2_fuselage_weight_analytical import ComputeFuselageMassAnalytical
from .a3_tail_weight import ComputeTailWeight
from .a3_tail_weight_gd import ComputeTailWeightGD
from .a3_tail_weight_torenbeek_gd import ComputeTailWeightTorenbeekGD
from .a4_flight_control_weight import ComputeFlightControlsWeight
from .a4_flight_control_weight_flops import ComputeFlightControlsWeightFLOPS
from .a5_landing_gear_weight import ComputeLandingGearWeight
from .a7_paint_weight import ComputePaintWeight
