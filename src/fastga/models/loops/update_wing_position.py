"""
Computation of wing position.
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

import numpy as np
import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain


@oad.RegisterOpenMDAOSystem("fastga.loop.wing_position", domain=ModelDomain.OTHER)
class UpdateWingPosition(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)
        self.add_input("data:handling_qualities:static_margin:target", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")

        self.add_output("data:geometry:wing:MAC:at25percent:x", units="m", val=3.5)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]
        target_static_margin = inputs["data:handling_qualities:static_margin:target"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_ratio = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        cg_x = inputs["data:weight:aircraft:CG:aft:x"]

        mac_position = (
            cg_x
            + 0.25 * l0_wing
            - cg_ratio * l0_wing
            - (static_margin - target_static_margin) * l0_wing
        )

        outputs["data:geometry:wing:MAC:at25percent:x"] = mac_position

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]
        target_static_margin = inputs["data:handling_qualities:static_margin:target"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_ratio = inputs["data:weight:aircraft:CG:aft:MAC_position"]

        partials[
            "data:geometry:wing:MAC:at25percent:x",
            "data:handling_qualities:stick_fixed_static_margin",
        ] = -l0_wing
        partials[
            "data:geometry:wing:MAC:at25percent:x", "data:handling_qualities:static_margin:target"
        ] = l0_wing
        partials["data:geometry:wing:MAC:at25percent:x", "data:geometry:wing:MAC:length"] = (
            0.25 - cg_ratio - (static_margin - target_static_margin)
        )
        partials[
            "data:geometry:wing:MAC:at25percent:x", "data:weight:aircraft:CG:aft:MAC_position"
        ] = -l0_wing
        partials["data:geometry:wing:MAC:at25percent:x", "data:weight:aircraft:CG:aft:x"] = 1.0
