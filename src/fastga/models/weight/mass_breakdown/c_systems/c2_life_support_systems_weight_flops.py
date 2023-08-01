"""
Estimation of life support systems weight.
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

import openmdao.api as om
import fastoad.api as oad

from .constants import SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS

from ..c_systems import (
    ComputeAirConditioningSystemsWeightFLOPS,
    ComputeAntiIcingSystemsWeightFLOPS,
    ComputeFixedOxygenSystemsWeightFLOPS,
)


@oad.RegisterSubmodel(
    SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS,
    "fastga.submodel.weight.mass.system.life_support_system.flops",
)
class ComputeLifeSupportSystemsWeightFLOPS(om.Group):
    """
    Weight estimation for life support systems

    This includes only air conditioning / pressurization, anti-icing and oxygen

    Insulation, internal lighting system, permanent security kits are neglected.
    Seats and installation of crew are already evaluated within d2_passenger_seats_weight

    Based on a statistical analysis. See :cite:`wells:2017` for de-icing and air conditioning and
    :cite:`roskampart5:1985` for the fixed oxygen weight.
    """

    def setup(self):

        self.add_subsystem(
            "c22_ac_system", ComputeAirConditioningSystemsWeightFLOPS(), promotes=["*"]
        )
        self.add_subsystem(
            "c23_anti_icing_system", ComputeAntiIcingSystemsWeightFLOPS(), promotes=["*"]
        )
        self.add_subsystem(
            "c26_fixed_oxygen", ComputeFixedOxygenSystemsWeightFLOPS(), promotes=["*"]
        )
