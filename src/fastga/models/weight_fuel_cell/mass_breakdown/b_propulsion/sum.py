"""Computation of the propulsion system mass."""
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

from fastoad.module_management.service_registry import RegisterSubmodel

from .b1_2_oil_weight import ComputeOilWeight
from .b1_ICengine_weight import ComputeEngineWeight
from .b2_fuel_lines_weight import ComputeFuelLinesWeight
from .b3_unusable_fuel_weight import ComputeUnusableFuelWeight
from .b4_Eengine_weight import ComputeEEngineWeight
from .b5_cables_weight import ComputeCablesWeight
from .b6_power_electronics_weight import ComputePowerElecWeight
from .b7_propeller_weight import ComputePropellerWeight
# from .b8_battery_weight import ComputeBatteryWeight
# from .b9_fuel_cells_weight import ComputeFuelCellWeight
# from .b10_bop_weight import ComputeBoPWeight
from .b11_inverter_weight import ComputeInverterWeight
# from .b12_h2_storage_weight import ComputeH2StorageWeight
from fastga.models.weight.mass_breakdown.constants import SUBMODEL_PROPULSION_MASS
from .constants import (
SUBMODEL_PROPULSION_FUELCELL_MASS,
SUBMODEL_PROPULSION_H2STORAGE_MASS,
SUBMODEL_PROPULSION_BATTERY_MASS,
SUBMODEL_PROPULSION_BOP_MASS,
)


@RegisterSubmodel(SUBMODEL_PROPULSION_MASS, "fastga.submodel.weight.mass.propulsion.hybrid.fuelcell")
class PropulsionWeight(om.Group):
    """Computes mass of propulsion system."""

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        self.add_subsystem("power_electronics_weight", ComputePowerElecWeight(), promotes=["*"])
        self.add_subsystem("cable_weight", ComputeCablesWeight(), promotes=["*"])
        self.add_subsystem("propeller_weight", ComputePropellerWeight(), promotes=["*"])
        self.add_subsystem("electric_engine_weight", ComputeEEngineWeight(propulsion_id=self.options["propulsion_id"]),
                           promotes=["*"])
        self.add_subsystem("battery_weight", RegisterSubmodel.get_submodel(SUBMODEL_PROPULSION_BATTERY_MASS), promotes=["*"])
        self.add_subsystem("fuel_cells_weight", RegisterSubmodel.get_submodel(SUBMODEL_PROPULSION_FUELCELL_MASS), promotes=["*"])
        self.add_subsystem("balance_of_plant_weight", RegisterSubmodel.get_submodel(SUBMODEL_PROPULSION_BOP_MASS), promotes=["*"])
        self.add_subsystem("inverter_weight", ComputeInverterWeight(), promotes=["*"])
        self.add_subsystem("h2_storage_weight", RegisterSubmodel.get_submodel(SUBMODEL_PROPULSION_H2STORAGE_MASS), promotes=["*"])


        hybrid_powertrain_sum = om.AddSubtractComp()

        hybrid_powertrain_sum.add_equation(
            "data:weight:propulsion:mass",
            [
                "data:weight:hybrid_powertrain:engine:mass",
                "data:weight:hybrid_powertrain:fuel_cell:mass",
                "data:weight:hybrid_powertrain:battery:mass",
                "data:weight:hybrid_powertrain:bop:total_mass",
                "data:weight:hybrid_powertrain:inverter:mass",
                "data:weight:hybrid_powertrain:h2_storage:mass",
            ],
            units="kg",
            desc="Mass of the hybrid propulsion system",
        )
        self.add_subsystem(
            "hybrid_powertrain_weight_sum", hybrid_powertrain_sum, promotes=["*"],
        )
