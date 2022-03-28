"""
Estimation of battery weight
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
from fastga.models.hybrid_powertrain.components.resources.constants import CELL_WEIGHT_FRACTION
from fastoad.module_management.service_registry import RegisterSubmodel
from .constants import SUBMODEL_PROPULSION_BATTERY_MASS

@RegisterSubmodel(SUBMODEL_PROPULSION_BATTERY_MASS, "fastga.submodel.weight.mass.propulsion.hybrid.fuelcell.battery.legacy")
class ComputeBatteryWeight(ExplicitComponent):
    """
    Weight estimation for battery
    Based on Zhao, Tianyuan, "Propulsive Battery Packs Sizing for Aviation Applications" (2018). Dissertations and
    Theses. 393. (https://commons.erau.edu/edt/393)
    """

    def setup(self):

        self.add_input("data:geometry:hybrid_powertrain:battery:N_series", val=np.nan, units=None)
        self.add_input("data:geometry:hybrid_powertrain:battery:N_parallel", val=np.nan, units=None)
        self.add_input("data:geometry:hybrid_powertrain:battery:cell_mass", val=np.nan, units="kg")

        self.add_output("data:weight:hybrid_powertrain:battery:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        N_series = inputs["data:geometry:hybrid_powertrain:battery:N_series"]
        N_parallel = inputs["data:geometry:hybrid_powertrain:battery:N_parallel"]
        cell_mass = inputs["data:geometry:hybrid_powertrain:battery:cell_mass"]

        b8 = N_series * N_parallel * cell_mass / CELL_WEIGHT_FRACTION

        outputs["data:weight:hybrid_powertrain:battery:mass"] = b8