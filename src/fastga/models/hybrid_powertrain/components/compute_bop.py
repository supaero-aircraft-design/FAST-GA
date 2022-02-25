""" Module that computes the Balance of Plant (including electrical, water cooling and fuelling subsystems besides of
H2 tanks and air compressor) for the FC-B hybrid propulsion model. """

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
# from .resources.constants import LC_SS_VOLUME_FRACTION, FC_OVERHEAD


class ComputeBoP(om.ExplicitComponent):
    """
    BoP is sized as an extension of the fuel cell stack ; therefore, the returned dimensions are the extra-length,
    width and height to be added to the dimensions of each fuel cell stack.
    BoP usually consists of :
        - Fuel cell stacks electrochemical subsystem (pressure regulators, tubing for gas, casing)
        - Air delivery system (compressors and air guidance tubes)
        - Water management system (water tank, pumps, filter and flow meter)
        - Heat-exchanger (HEX) system
        - Electrical and electronic support system and control and internal battery subsystem
    The HEX system and the compressor(s) are computed outside of this discipline which is why a fitting parameter has
    been added in input to adjust the output dimensions.

    """

    def setup(self):
        # Input dimensions refer to a single fuel cell stack if there are more than one.
        # self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_volume", val=np.nan, units='L')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_length", val=0.490, units='m')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_width", val=0.155, units='m')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_height", val=np.nan, units='m')
        # self.add_input("data:geometry:hybrid_powertrain:compressor:volume", val=np.nan, units='L')
        # self.add_input("data:geometry:hybrid_powertrain:bop:pressure_reg_volume", val=1.3, units='L')
        self.add_input("data:geometry:hybrid_powertrain:bop:fitting_factor", val=1, units='L',
                       desc='Scale factor to account for the subsystems computed outside of this discipline')

        # self.add_output("data:geometry:hybrid_powertrain:bop:fc_ss_volume", units='L')
        self.add_output("data:geometry:hybrid_powertrain:bop:extra_length", units='m')
        self.add_output("data:geometry:hybrid_powertrain:bop:extra_width", units='m')
        self.add_output("data:geometry:hybrid_powertrain:bop:extra_height", units='m')

        self.declare_partials('*', '*', method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # stack_volume = inputs['data:geometry:hybrid_powertrain:fuel_cell:stack_volume']
        stack_length = inputs['data:geometry:hybrid_powertrain:fuel_cell:stack_length']
        stack_width = inputs['data:geometry:hybrid_powertrain:fuel_cell:stack_width']
        stack_height = inputs['data:geometry:hybrid_powertrain:fuel_cell:stack_height']
        # compressor_volume = inputs['data:geometry:hybrid_powertrain:compressor:volume']
        # pressure_reg_volume = inputs['data:geometry:hybrid_powertrain:bop:pressure_reg_volume']
        FITTING_FACTOR = inputs['data:geometry:hybrid_powertrain:bop:fitting_factor']

        # Computing BoP dimensions based on stack dimensions.
        # Since the compressor(s) and the heat exchanger are computed outside of the BoP and considering ratios between
        # the dimensions of the PowerCellution V Stack sized for 30 kW and the Power Generation System 30, we assume
        # that for the whole FC system, the BoP :
        #     - increases length by 136 %
        #     - increases width by 298 %
        #     - increases height by 173 %
        # Based on a comparison between the two following fuel cell stacks :
        #     - https://www.datocms-assets.com/36080/1611437781-v-stack.pdf
        #     - https://www.datocms-assets.com/36080/1636022163-power-generation-system-30-v221.pdf

        L = 0.36 * stack_length * FITTING_FACTOR
        l = 1.98 * stack_width * FITTING_FACTOR
        h = 0.73 * stack_height * FITTING_FACTOR

        outputs['data:geometry:hybrid_powertrain:bop:extra_length'] = L
        outputs['data:geometry:hybrid_powertrain:bop:extra_width'] = l
        outputs['data:geometry:hybrid_powertrain:bop:extra_height'] = h


        ### Unused method
        # Determining FC system overhead volume. That includes :
        #     - power cables
        #     - tubing for gas and liquid flows
        #     - casing and/or structural support
        #     - sensors
        #     - miscellaneous cooling equipments
        # FC overhead is included in the BoP of the FC system and its volume is only provided for information.
        #
        # fc_overhead_volume = FC_OVERHEAD * (stack_volume + compressor_volume + pressure_reg_volume)
        # outputs['data:geometry:hybrid_powertrain:bop:fc_ss_volume'] = fc_overhead_volume