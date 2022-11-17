""" Estimation of battery(ies) center of gravity """

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


class ComputeBatteryCG(ExplicitComponent):
    """
    Center of gravity estimation of the battery considering a maximum of 2 battery packs.

    If there is only 1 pack : assuming an engine configuration with the battery pack behind the pilot seat and H2 tanks.
    Based on : 'Boeing Fuel Cell Demonstrator' - https://arc.aiaa.org/doi/10.2514/1.42234

    If there are 2 : assuming an engine configuration with both batteries at the front of the aircraft (just behind the
    fuel cell stacks).
    Based on : 'ENFICA-FC: Design of transport aircraft powered by fuel cell & flight test of zero emission 2-seater
    aircraft powered by fuel cells fueled by hydrogen' - G. Romeo*, F. Borello, G. Correa, E. Cestino
    """

    def setup(self):
        self.add_input("data:geometry:hybrid_powertrain:battery:nb_packs", val=np.nan, units=None)
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:number_stacks", val=np.nan)
        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units='m')
        self.add_input("data:geometry:hybrid_powertrain:battery:pack_volume", val=np.nan, units='m**3')
        self.add_input("data:geometry:propulsion:nacelle:master_cross_section", val=np.nan, units='m**2')

        self.add_output("data:weight:hybrid_powertrain:battery:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nb_stacks = inputs["data:geometry:hybrid_powertrain:fuel_cell:number_stacks"]

        if nb_stacks > 1:
            engine_cg = inputs["data:weight:propulsion:engine:CG:x"]
            nac_section = inputs["data:geometry:propulsion:nacelle:master_cross_section"]
            pack_vol = inputs["data:geometry:hybrid_powertrain:battery:pack_volume"]
            cg_b4 = engine_cg + pack_vol/(nac_section/2)/2
        else:

            nb_packs = inputs['data:geometry:hybrid_powertrain:battery:nb_packs']
            fus_length = inputs["data:geometry:fuselage:length"]
            fus_front_length = inputs["data:geometry:fuselage:front_length"]
            cabin_length = inputs["data:geometry:cabin:length"]

            if nb_packs == 1:  # Single battery placed behind the pilot seat and the hydrogen tanks : 95% of the cabin
                cg_b4 = fus_front_length + 0.95 * cabin_length
            else:  # Both batteries placed one after the other at the front of the aircraft behind fuel cell stacks
                cg_batt_1 = 0.7 * fus_front_length  # 70% of the fuselage front
                cg_batt_2 = 0.85 * fus_front_length  # 85% of the fuselage front
                cg_b4 = (cg_batt_1 + cg_batt_2) / 2

        outputs["data:weight:hybrid_powertrain:battery:CG:x"] = cg_b4
