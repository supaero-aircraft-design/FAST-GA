"""Package containing the subcomponents necessary for the analytical mass estimation."""

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

from .compute_web_mass import ComputeWebMass
from .compute_lower_flange import ComputeLowerFlange
from .compute_upper_flange import ComputeUpperFlange
from .compute_skin_mass import ComputeSkinMass
from .compute_misc_mass import ComputeMiscMass
from .compute_ribs_mass import ComputeRibsMass
from .compute_primary_mass import ComputePrimaryMass
from .compute_secondary_mass import ComputeSecondaryMass
from .update_wing_mass import UpdateWingMass
