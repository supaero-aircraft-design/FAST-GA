"""
    Estimation of maximum center of gravity ratio.
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
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group


class ComputeMaxMinCGRatio(Group):
    # TODO: Document equations. Cite sources
    """Extrema center of gravity ratio estimation"""

    def setup(self):

        self.add_subsystem("comp_cg_aft_mac", ComputeAftCGMac(), promotes=["*"])
        self.add_subsystem("comp_cg_fwd_mac", ComputeFwdCGMac(), promotes=["*"])
        self.add_subsystem("comp_cg_aft_x", ComputeAftCGX(), promotes=["*"])

        # Comment this line if ComputeGlobalCG is used
        self.add_subsystem("comp_cg_fwd_x", ComputeFwdCGX(), promotes=["*"])


class ComputeAftCGMac(ExplicitComponent):
    """Aft center of gravity MAC position estimation"""

    def setup(self):

        self.add_input("data:weight:aircraft:CG:flight_condition:max:MAC_position", val=np.nan)
        self.add_input("data:weight:aircraft:CG:ground_condition:max:MAC_position", val=np.nan)
        self.add_input(
            "settings:weight:aircraft:CG:aft:MAC_position:margin",
            val=0.05,
            desc="Added margin for getting most aft CG position, "
            "as ratio of mean aerodynamic chord",
        )

        self.add_output("data:weight:aircraft:CG:aft:MAC_position")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ground_conditions_aft = inputs["data:weight:aircraft:CG:ground_condition:max:MAC_position"]
        flight_conditions_aft = inputs["data:weight:aircraft:CG:flight_condition:max:MAC_position"]
        margin_aft = inputs["settings:weight:aircraft:CG:aft:MAC_position:margin"]

        cg_max_aft_mac = max(ground_conditions_aft, flight_conditions_aft) + margin_aft

        outputs["data:weight:aircraft:CG:aft:MAC_position"] = cg_max_aft_mac


class ComputeFwdCGMac(ExplicitComponent):
    """Forward center of gravity MAC position estimation"""

    def setup(self):

        self.add_input("data:weight:aircraft:CG:flight_condition:min:MAC_position", val=np.nan)
        self.add_input("data:weight:aircraft:CG:ground_condition:min:MAC_position", val=np.nan)
        self.add_input("settings:weight:aircraft:CG:range", val=np.nan)
        self.add_input(
            "settings:weight:aircraft:CG:fwd:MAC_position:margin",
            val=0.03,
            desc="Added margin for getting most fwd CG position, "
            "as ratio of mean aerodynamic chord",
        )
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)

        self.add_output("data:weight:aircraft:CG:fwd:MAC_position")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ground_conditions_fwd = inputs["data:weight:aircraft:CG:ground_condition:min:MAC_position"]
        flight_conditions_fwd = inputs["data:weight:aircraft:CG:flight_condition:min:MAC_position"]
        cg_range = inputs["settings:weight:aircraft:CG:range"]
        margin_fwd = inputs["settings:weight:aircraft:CG:fwd:MAC_position:margin"]
        cg_max_aft_mac = inputs["data:weight:aircraft:CG:aft:MAC_position"]

        cg_min_fwd_mac = (
            min(ground_conditions_fwd, flight_conditions_fwd, cg_max_aft_mac - cg_range)
            - margin_fwd
        )

        outputs["data:weight:aircraft:CG:fwd:MAC_position"] = cg_min_fwd_mac


class ComputeAftCGX(ExplicitComponent):
    """Aft center of gravity x position estimation"""

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)

        self.add_output("data:weight:aircraft:CG:aft:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        mac_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        cg_max_aft_mac = inputs["data:weight:aircraft:CG:aft:MAC_position"]

        outputs["data:weight:aircraft:CG:aft:x"] = (
            mac_position - 0.25 * l0_wing + cg_max_aft_mac * l0_wing
        )


class ComputeFwdCGX(ExplicitComponent):
    """Forward center of gravity x position estimation"""

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:MAC_position", val=np.nan)

        self.add_output("data:weight:aircraft:CG:fwd:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        mac_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        cg_min_fwd_mac = inputs["data:weight:aircraft:CG:fwd:MAC_position"]

        outputs["data:weight:aircraft:CG:fwd:x"] = (
            mac_position - 0.25 * l0_wing + cg_min_fwd_mac * l0_wing
        )
