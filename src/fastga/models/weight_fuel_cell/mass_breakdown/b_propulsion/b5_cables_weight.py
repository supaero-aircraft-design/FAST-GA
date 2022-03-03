"""
Estimation of cables weight
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
# import warnings

class ComputeCablesWeight(ExplicitComponent):
    """
    Weight estimation for cables
    Based on : 1. Aircraft Wires & Cables
    2. Parametric modeling of the aircraft electrical supply system for overall
    conceptual systems design - T Bielsky, M Junemann, F Thielecke
    Based on a model found in FAST-GA-ELEC.
    """

    def setup(self):

        self.add_input("data:propulsion:hybrid_powertrain:cable:lsw", val=np.nan, units="kg/km")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        self.add_output("data:weight:hybrid_powertrain:cable:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        length_spec_wt = inputs["data:propulsion:hybrid_powertrain:cable:lsw"]
        cabin_length = inputs["data:geometry:cabin:length"]

        b5 = 2 * 2.20462 * length_spec_wt * cabin_length / 1000 # mass has been multiplied by 2
        # to account for redundancy, this factor can be reduced as per the convenience of the designer

        outputs["data:weight:hybrid_powertrain:cable:mass"] = b5
