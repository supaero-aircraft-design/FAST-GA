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

from .constants import SUBMODEL_AIRCRAFT_FORWARD_CG_MAC


@oad.RegisterSubmodel(
    SUBMODEL_AIRCRAFT_FORWARD_CG_MAC, "fastga.submodel.weight.cg.aircraft.most_forward.mac.legacy"
)
class ComputeForwardCGMac(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Extreme (most forward) center of gravity ratio estimation in MAC"""

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

        self.declare_partials(of="*", wrt="*", method="exact")

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

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ground_conditions_fwd = inputs["data:weight:aircraft:CG:ground_condition:min:MAC_position"]
        flight_conditions_fwd = inputs["data:weight:aircraft:CG:flight_condition:min:MAC_position"]
        cg_range = inputs["settings:weight:aircraft:CG:range"]
        cg_max_aft_mac = inputs["data:weight:aircraft:CG:aft:MAC_position"]

        min_cg = min(ground_conditions_fwd, flight_conditions_fwd, cg_max_aft_mac - cg_range)

        if min_cg == ground_conditions_fwd:
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:ground_condition:min:MAC_position",
            ] = 1.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:flight_condition:min:MAC_position",
            ] = 0.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position", "settings:weight:aircraft:CG:range"
            ] = 0.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:aft:MAC_position",
            ] = 0.0
        elif min_cg == flight_conditions_fwd:
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:ground_condition:min:MAC_position",
            ] = 0.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:flight_condition:min:MAC_position",
            ] = 1.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position", "settings:weight:aircraft:CG:range"
            ] = 0.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:aft:MAC_position",
            ] = 0.0
        else:
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:ground_condition:min:MAC_position",
            ] = 0.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:flight_condition:min:MAC_position",
            ] = 0.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position", "settings:weight:aircraft:CG:range"
            ] = -1.0
            partials[
                "data:weight:aircraft:CG:fwd:MAC_position",
                "data:weight:aircraft:CG:aft:MAC_position",
            ] = 1.0

        partials[
            "data:weight:aircraft:CG:fwd:MAC_position",
            "settings:weight:aircraft:CG:fwd:MAC_position:margin",
        ] = -1.0
