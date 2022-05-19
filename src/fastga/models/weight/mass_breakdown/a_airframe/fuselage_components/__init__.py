"""Fuselage component computation."""
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

from .compute_additional_bending_material_mass_h import ComputeAddBendingMassHorizontal
from .compute_additional_bending_material_mass_v import ComputeAddBendingMassVertical
from .compute_bulkhead_mass import ComputeBulkhead
from .compute_cone_mass import ComputeTailCone
from .compute_doors_mass import ComputeDoors
from .compute_engine_support_mass import ComputeEngineSupport
from .compute_floor_mass import ComputeFloor
from .compute_insulation_mass import ComputeInsulation
from .compute_nlg_hatch_mass import ComputeNLGHatch
from .compute_shell_mass import ComputeShell
from .compute_windows_mass import ComputeWindows
from .compute_wing_fuselage_connection_mass import ComputeWingFuselageConnection
