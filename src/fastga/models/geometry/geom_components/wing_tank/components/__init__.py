"""
Python package that contains the module for the computation of the MFW with the advanced method.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

from .compute_wing_tank_span import ComputeWingTankSpans
from .compute_wing_tank_chord_array import ComputeWingTankChordArray
from .compute_wing_tank_t_c_array import ComputeWingTankRelativeThicknessArray
from .compute_wing_tank_thickness_array import ComputeWingTankThicknessArray
from .compute_wing_tank_reduced_width_array import ComputeWingTankReducedWidthArray
from .compute_wing_tank_cross_section import ComputeWingTankCrossSectionArray
from .compute_mfw_from_wing_tanks_capacity import ComputeMFWFromWingTanksCapacity
from .compute_wing_tank_width_array import ComputeWingTankWidthArray
from .compute_wing_tank_y_array import ComputeWingTankYArray
from .compute_wing_tanks_capacity import ComputeWingTanksCapacity
