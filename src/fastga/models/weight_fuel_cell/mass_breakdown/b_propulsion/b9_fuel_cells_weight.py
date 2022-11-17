"""
Estimation of fuel cell stacks weight
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
from fastoad.module_management.service_registry import RegisterSubmodel
from .constants import SUBMODEL_PROPULSION_FUELCELL_MASS

@RegisterSubmodel(SUBMODEL_PROPULSION_FUELCELL_MASS, 'fastga.submodel.weight.mass.propulsion.hybrid.fuelcell.legacy')
class ComputeFuelCellWeight(ExplicitComponent):
    """
    Weight estimation for fuel cells stacks.
    """

    def setup(self):

        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:number_cells", val=np.nan, units=None)
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:stack_area", val=np.nan, units="cm**2")
        self.add_input("data:geometry:hybrid_powertrain:fuel_cell:number_stacks", val=np.nan, units=None)

        self.add_output("data:weight:hybrid_powertrain:fuel_cell:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Computes the weight of the fuel cell stack(s) given the total number of cells.
        It is assumed that weight can be described as a linear function of the number of cells with the parameters :
            a = 0.1028153153153153
            b = 8.762162162162165
        Those parameters are based on data retrieved from the PowerCellution V Stack fuel cell :
            https://www.datocms-assets.com/36080/1611437781-v-stack.pdf.
        """
        # The fuel cell of PowerCellution V stack being a ref 759.5cm**2, a ratio of area is applied
        nb_stacks = inputs["data:geometry:hybrid_powertrain:fuel_cell:number_stacks"]
        cell_area = inputs["data:geometry:hybrid_powertrain:fuel_cell:stack_area"]
        a_ratio = cell_area / 759.5

        cell_number = inputs['data:geometry:hybrid_powertrain:fuel_cell:number_cells'] * nb_stacks
        b9 = (cell_number * 0.103 + 8.762 ) * a_ratio  # [kg]

        outputs['data:weight:hybrid_powertrain:fuel_cell:mass'] = b9