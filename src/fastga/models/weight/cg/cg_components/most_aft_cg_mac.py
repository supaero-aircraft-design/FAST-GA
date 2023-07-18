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
import openmdao.api as om
import fastoad.api as oad

from .constants import SUBMODEL_AIRCRAFT_AFT_CG_MAC


@oad.RegisterSubmodel(
    SUBMODEL_AIRCRAFT_AFT_CG_MAC, "fastga.submodel.weight.cg.aircraft.most_aft.mac.legacy"
)
class ComputeAftCGMac(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Extreme (most aft) center of gravity ratio estimation in MAC"""

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

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ground_conditions_aft = inputs["data:weight:aircraft:CG:ground_condition:max:MAC_position"]
        flight_conditions_aft = inputs["data:weight:aircraft:CG:flight_condition:max:MAC_position"]
        margin_aft = inputs["settings:weight:aircraft:CG:aft:MAC_position:margin"]

        cg_max_aft_mac = max(ground_conditions_aft, flight_conditions_aft) + margin_aft

        outputs["data:weight:aircraft:CG:aft:MAC_position"] = cg_max_aft_mac

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ground_conditions_aft = inputs["data:weight:aircraft:CG:ground_condition:max:MAC_position"]
        flight_conditions_aft = inputs["data:weight:aircraft:CG:flight_condition:max:MAC_position"]

        partials[
            "data:weight:aircraft:CG:aft:MAC_position",
            "settings:weight:aircraft:CG:aft:MAC_position:margin",
        ] = 1.0

        if ground_conditions_aft > flight_conditions_aft:
            partials[
                "data:weight:aircraft:CG:aft:MAC_position",
                "data:weight:aircraft:CG:ground_condition:max:MAC_position",
            ] = 1.0
            partials[
                "data:weight:aircraft:CG:aft:MAC_position",
                "data:weight:aircraft:CG:flight_condition:max:MAC_position",
            ] = 0.0
        else:
            partials[
                "data:weight:aircraft:CG:aft:MAC_position",
                "data:weight:aircraft:CG:ground_condition:max:MAC_position",
            ] = 0.0
            partials[
                "data:weight:aircraft:CG:aft:MAC_position",
                "data:weight:aircraft:CG:flight_condition:max:MAC_position",
            ] = 1.0
