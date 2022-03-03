"""
Estimation of propeller weight for an electric engine
"""
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
from openmdao.core.explicitcomponent import ExplicitComponent


class ComputePropellerWeight(ExplicitComponent):
    """
    Weight estimation for propeller
    Based on FAST-GA-ELEC and : Airplane Design, Dr. Jan Roskam, Part 5 (Page 1537)
    """
    def setup(self):

        self.add_input("data:geometry:propeller:prop_number", val=np.nan, units=None)
        self.add_input("data:geometry:propeller:blades_number", val=np.nan, units=None)
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="ft")
        self.add_input("data:propulsion:HE_engine:max_power", units="W")
        self.add_input("data:geometry:propulsion:count", val=np.nan, units=None)
        self.add_input("settings:weight:hybrid_powertrain:prop_reduction_factor", val=np.nan, units=None)

        self.add_output("data:weight:hybrid_powertrain:propeller:mass", units="lb")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        num_prop = inputs["data:geometry:propeller:prop_number"]
        num_blades = inputs["data:geometry:propeller:blades_number"]
        prop_diameter = inputs["data:geometry:propeller:diameter"]
        max_power = inputs["data:propulsion:HE_engine:max_power"]
        num_eng = inputs["data:geometry:propulsion:count"]
        k_factor_prop = inputs["settings:weight:hybrid_powertrain:prop_reduction_factor"]

        b7 = k_factor_prop * 31.92 * num_prop * (num_blades ** 0.391) * (
                (prop_diameter * (max_power*0.00134102/num_eng)/1000) ** 0.782)

        outputs["data:weight:hybrid_powertrain:propeller:mass"] = b7
