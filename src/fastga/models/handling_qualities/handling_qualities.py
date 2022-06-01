"""
Estimation of static margin.
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
from fastoad.module_management.constants import ModelDomain

from fastga.models.aerodynamics.aero_center import ComputeAeroCenter
from fastga.models.handling_qualities.compute_static_margin import _ComputeStaticMargin
from fastga.models.handling_qualities.tail_sizing.compute_to_rotation_limit import (
    ComputeTORotationLimitGroup,
)
from fastga.models.handling_qualities.tail_sizing.compute_balked_landing_limit import (
    ComputeBalkedLandingLimit,
)


@oad.RegisterOpenMDAOSystem(
    "fastga.handling_qualities.all_handling_qualities", domain=ModelDomain.HANDLING_QUALITIES
)
class ComputeHandlingQualities(om.Group):
    """
    Calculate static margins and maneuver limits.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem("aero_center", ComputeAeroCenter(), promotes=["*"])
        self.add_subsystem("static_margin", _ComputeStaticMargin(), promotes=["*"])
        self.add_subsystem(
            "to_rotation_limit",
            ComputeTORotationLimitGroup(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "balked_landing_limit",
            ComputeBalkedLandingLimit(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
