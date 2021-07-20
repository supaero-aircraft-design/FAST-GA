"""Estimation of static margin"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

import numpy as np
import openmdao.api as om

from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain

from fastga.models.aerodynamics.aero_center import ComputeAeroCenter


@RegisterOpenMDAOSystem(
    "fastga.handling_qualities.static_margin", domain=ModelDomain.HANDLING_QUALITIES
)
class ComputeStaticMargin(om.Group):
    """
    Calculate aero-center and global static margin
    """

    def setup(self):
        self.add_subsystem("aero_center", ComputeAeroCenter(), promotes=["*"])
        self.add_subsystem("static_margin", _ComputeStaticMargin(), promotes=["*"])


class _ComputeStaticMargin(om.ExplicitComponent):
    """
    Computation of static margin i.e. difference between CG ratio and neutral
    point.
    """

    def initialize(self):
        self.options.declare("target", types=float, allow_none=True, default=None)

    def setup(self):
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input("data:aerodynamics:cruise:neutral_point:stick_fixed:x", val=np.nan)
        self.add_input("data:aerodynamics:cruise:neutral_point:stick_free:x", val=np.nan)

        self.add_output("data:handling_qualities:stick_fixed_static_margin")
        self.add_output("data:handling_qualities:stick_free_static_margin")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cg_ratio = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        ac_ratio_fixed = inputs["data:aerodynamics:cruise:neutral_point:stick_fixed:x"]
        ac_ratio_free = inputs["data:aerodynamics:cruise:neutral_point:stick_free:x"]

        outputs["data:handling_qualities:stick_fixed_static_margin"] = ac_ratio_fixed - cg_ratio
        outputs["data:handling_qualities:stick_free_static_margin"] = ac_ratio_free - cg_ratio
