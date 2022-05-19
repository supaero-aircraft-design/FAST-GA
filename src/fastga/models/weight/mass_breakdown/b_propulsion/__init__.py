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

from .b1_2_oil_weight import ComputeOilWeight
from .b1_engine_weight import ComputeEngineWeight, ComputeEngineWeightRaymer
from .b2_fuel_lines_weight import ComputeFuelLinesWeight, ComputeFuelLinesWeightFLOPS
from .b3_unusable_fuel_weight import ComputeUnusableFuelWeight
