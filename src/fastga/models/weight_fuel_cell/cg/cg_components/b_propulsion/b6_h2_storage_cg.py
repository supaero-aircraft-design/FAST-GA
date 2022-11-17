""" Estimation of H2 tanks center of gravity """

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
from fastoad.module_management.service_registry import RegisterSubmodel
from openmdao.core.explicitcomponent import ExplicitComponent
from fastga.models.weight.cg.cg_components.constants import SUBMODEL_TANK_CG


@RegisterSubmodel(SUBMODEL_TANK_CG, "fastga.submodel.weight.cg.propulsion.tank.gh2")
class ComputeH2StorageCG(ExplicitComponent):
    """
    Center of gravity estimation of the hydrogen tanks considering a maximum of 2.
    Assuming hydrogen is stored behind the pilot seat, in height in case there are more than 1 (meaning CG.x for several
    tanks is the same as for one)
    """

    def setup(self):
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:number_stacks", val=np.nan)
        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units='m')
        self.add_input("data:geometry:hybrid_powertrain:bop:length_in_nacelle", val=np.nan, units='m')
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_height", val=np.nan, units='m')
        self.add_input("data:geometry:hybrid_powertrain:h2_storage:tank_ext_length", val=np.nan, units='m')
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:hybrid_powertrain:battery:pack_volume", val=np.nan, units='m**3')
        self.add_input("data:geometry:propulsion:nacelle:master_cross_section", val=np.nan, units='m**2')

        self.add_output("data:weight:propulsion:tank:CG:x", units="m")
        self.add_output("data:weight:hybrid_powertrain:h2_storage:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nb_stacks = inputs["data:geometry:hybrid_powertrain:fuel_cell:number_stacks"]

        if nb_stacks > 1 :
            engine_cg = inputs["data:weight:propulsion:engine:CG:x"]
            bop_length = inputs["data:geometry:hybrid_powertrain:bop:length_in_nacelle"]
            fc_height = inputs["data:geometry:hybrid_powertrain:fuel_cell:stack_height"]
            tank_length = inputs["data:geometry:hybrid_powertrain:h2_storage:tank_ext_length"]
            nac_section = inputs["data:geometry:propulsion:nacelle:master_cross_section"]
            pack_vol = inputs["data:geometry:hybrid_powertrain:battery:pack_volume"]

            b6 = engine_cg + pack_vol / (nac_section/2) + bop_length + fc_height + tank_length/2
        else:
            fus_front_length = inputs["data:geometry:fuselage:front_length"]
            cabin_length = inputs["data:geometry:cabin:length"]

            # H2 tanks assumed to be placed just behind the pilot seat (stacked in height if more than 1)
            b6 = fus_front_length + 0.5 * cabin_length  # 110% of the cabin length

        outputs["data:weight:propulsion:tank:CG:x"] = b6 # This is for loadcases with fuel weight
        outputs["data:weight:hybrid_powertrain:h2_storage:CG:x"] = b6 # this is for empty CG
