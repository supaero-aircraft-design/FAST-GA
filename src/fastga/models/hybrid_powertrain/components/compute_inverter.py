""" Module that computes the inverter-motor controller in a hybrid propulsion model (FC-B configuration). """

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

import openmdao.api as om
import numpy as np
from .resources.constants import AF


class ComputeInverter(om.ExplicitComponent):
    """
    Sizing the inverter based on the method used here :
        https://electricalnotes.wordpress.com/2015/10/02/calculate-size-of-inverter-battery-bank/
    Default value for efficiency is set at 94%.
    """
    def setup(self):

        self.add_input("data:propulsion:hybrid_powertrain:fuel_cell:design_power", val=np.nan, units='W')
        self.add_input("data:propulsion:hybrid_powertrain:battery:max_power", val=np.nan, units='W')
        self.add_input("data:propulsion:hybrid_powertrain:inverter:power_density", val=0.070, units='kW/m**3')
        self.add_input("data:propulsion:hybrid_powertrain:inverter:efficiency", val=0.94, units=None)
        self.add_input("data:geometry:hybrid_powertrain:inverter:length_width_ratio", val=2, units=None)
        self.add_input("data:geometry:hybrid_powertrain:inverter:length_height_ratio", val=4, units=None)

        self.add_output("data:propulsion:hybrid_powertrain:inverter:output_power", units='W')
        self.add_output("data:geometry:hybrid_powertrain:inverter:volume", units='m**3')
        self.add_output("data:geometry:hybrid_powertrain:inverter:length", units='m')
        self.add_output("data:geometry:hybrid_powertrain:inverter:width", units='m')
        self.add_output("data:geometry:hybrid_powertrain:inverter:height", units='m')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fc_max_power = inputs['data:propulsion:hybrid_powertrain:fuel_cell:design_power']
        batt_max_power = inputs['data:propulsion:hybrid_powertrain:battery:max_power']
        power_density = inputs['data:propulsion:hybrid_powertrain:inverter:power_density']
        eff = inputs['data:propulsion:hybrid_powertrain:inverter:efficiency']
        L_to_l_ratio = inputs['data:geometry:hybrid_powertrain:inverter:length_width_ratio']
        L_to_h_ratio = inputs['data:geometry:hybrid_powertrain:inverter:length_height_ratio']

        max_elec_load = (fc_max_power + batt_max_power)
        des_power = max_elec_load * (1 + AF) / eff
        outputs['data:propulsion:hybrid_powertrain:inverter:output_power'] = des_power

        # Sizing the inverter as a parallelepiped of default dimensions ratios :
        #     Length = 2 * Width = 4 * Height
        # (Arbitrary values chosen considering common dimensions for inverters).

        vol = des_power / 1000 / power_density  # [m**3]
        L = (L_to_h_ratio * L_to_l_ratio * vol) ** (1/3)
        l = L / L_to_l_ratio
        h = L / L_to_h_ratio

        outputs['data:geometry:hybrid_powertrain:inverter:volume'] = vol
        outputs['data:geometry:hybrid_powertrain:inverter:length'] = L
        outputs['data:geometry:hybrid_powertrain:inverter:width'] = l
        outputs['data:geometry:hybrid_powertrain:inverter:height'] = h