"""
Estimation of inverter weight
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


class ComputeInverterWeight(ExplicitComponent):
    """
    Weight estimation for the inverter in a hybrid powertrain.
    """

    def setup(self):

        self.add_input("data:propulsion:hybrid_powertrain:inverter:specific_power", val=np.nan, units="W/kg")
        self.add_input("data:propulsion:hybrid_powertrain:inverter:output_power", val=np.nan, units="W")

        self.add_output("data:weight:hybrid_powertrain:inverter:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        spe_power = inputs['data:propulsion:hybrid_powertrain:inverter:specific_power']
        power = inputs['data:propulsion:hybrid_powertrain:inverter:output_power']

        b11 = power / spe_power

        outputs['data:weight:hybrid_powertrain:inverter:mass'] = b11
