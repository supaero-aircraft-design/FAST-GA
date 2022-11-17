""" Estimation of fuel cell stacks center of gravity """

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


class ComputeFuelCellCG(ExplicitComponent):
    """
    Center of gravity estimation of the fuel cells considering a maximum of two stacks.
    Assuming that if there are 2 stacks, they are placed next to each other (on an axis perpendicular to the fuselage).
    Therefore CG.x doesn't depend on the number of stacks (still an input in case it's modified).
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:number_stacks", val=np.nan)
        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units='m')
        self.add_input("data:geometry:hybrid_powertrain:bop:length_in_nacelle", val=np.nan, units='m')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_height", val=np.nan, units='m')
        self.add_input("data:geometry:hybrid_powertrain:battery:pack_volume", val=np.nan, units='m**3')
        self.add_input("data:geometry:propulsion:nacelle:master_cross_section", val=np.nan, units='m**2')

        self.add_output("data:weight:hybrid_powertrain:fuel_cell:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nb_stacks = inputs["data:geometry:hybrid_powertrain:fuel_cell:number_stacks"]

        if nb_stacks > 1:
            engine_cg = inputs["data:weight:propulsion:engine:CG:x"]
            bop_length = inputs["data:geometry:hybrid_powertrain:bop:length_in_nacelle"]
            fc_height = inputs["data:geometry:hybrid_powertrain:fuel_cell:stack_height"]
            nac_section = inputs["data:geometry:propulsion:nacelle:master_cross_section"]
            pack_vol = inputs["data:geometry:hybrid_powertrain:battery:pack_volume"]

            cg_b5 = engine_cg + pack_vol/ (nac_section/2) + (bop_length + fc_height)/2
        else:

            fus_front_length = inputs["data:geometry:fuselage:front_length"]

            # Fuel cell stack(s) assumed to be placed at 50% of the fuselage front length, in parallel if more than 1 stack
            cg_b5 = 0.5 * fus_front_length

        outputs["data:weight:hybrid_powertrain:fuel_cell:CG:x"] = cg_b5
