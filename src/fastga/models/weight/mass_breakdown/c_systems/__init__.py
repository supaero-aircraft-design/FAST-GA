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

from .c1_electric_power_systems_weight import ComputeElectricWeight
from .c5_hydraulic_power_systems_weight import ComputeHydraulicWeight
from .c22_air_conditioning_weight import ComputeAirConditioningSystemsWeight
from .c22_air_conditioning_weight_flops import ComputeAirConditioningSystemsWeightFLOPS
from .c23_de_icing_weight import ComputeAntiIcingSystemsWeight
from .c23_de_icing_weight_flops import ComputeAntiIcingSystemsWeightFLOPS
from .c26_fixed_oxygen_weight import ComputeFixedOxygenSystemsWeight
from .c26_fixed_oxygen_weight_flops import ComputeFixedOxygenSystemsWeightFLOPS
from .c3_avionics_systems_weight import ComputeAvionicsSystemsWeight
from .c3_avionics_systems_from_uninstalled_weight import ComputeAvionicsSystemsWeightFromUninstalled
from .c4_recording_systems_weight import ComputeRecordingSystemsWeight
from .systems_total_mass import ComputeSystemMass
