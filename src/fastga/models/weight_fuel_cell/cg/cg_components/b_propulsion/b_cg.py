"""Estimation of center of gravity for a fuel propulsion system."""
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

from .b1_engine_cg import ComputeEngineCG
from .b2_fuel_lines_cg import ComputeFuelLinesCG
# from .b3_tank_cg import ComputeTankCG
from .b4_battery_cg import ComputeBatteryCG
from .b5_fuel_cell_cg import ComputeFuelCellCG
# from .b6_h2_storage_cg import ComputeH2StorageCG

from fastoad.module_management.service_registry import RegisterSubmodel
from fastga.models.weight.cg.cg_components.constants import SUBMODEL_PROPULSION_CG


@RegisterSubmodel(SUBMODEL_PROPULSION_CG, "fastga.submodel.weight.cg.propulsion.hybrid.fuelcell")
class FuelPropulsionCG(om.Group):
    def setup(self):
        self.add_subsystem("engine_cg", ComputeEngineCG(), promotes=["*"])
        self.add_subsystem("fuel_lines_cg", ComputeFuelLinesCG(), promotes=["*"])
        self.add_subsystem("fuel_cells_cg", ComputeFuelCellCG(), promotes=["*"])
        self.add_subsystem("battery_cg", ComputeBatteryCG(), promotes=["*"])
        self.add_subsystem("propulsion_cg", ComputeFuelPropulsionCG(), promotes=["*"])


class ComputeFuelPropulsionCG(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:propulsion:engine:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:propulsion:fuel_lines:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:hybrid_powertrain:fuel_cell:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:hybrid_powertrain:battery:CG:x", units="m", val=np.nan)
        self.add_input("data:weight:hybrid_powertrain:engine:mass", units="kg", val=np.nan)
        self.add_input("data:weight:propulsion:fuel_lines:mass", units="kg", val=np.nan)
        self.add_input("data:weight:hybrid_powertrain:fuel_cell:mass", units="kg", val=np.nan)
        self.add_input("data:weight:hybrid_powertrain:battery:mass", units="kg", val=np.nan)
        # self.add_input("data:weight:propulsion:mass", units="kg", val=np.nan)

        self.add_output("data:weight:propulsion:CG:x", units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        engine_cg = inputs["data:weight:propulsion:engine:CG:x"]
        fuel_lines_cg = inputs["data:weight:propulsion:fuel_lines:CG:x"]
        fuel_cell_cg = inputs=["data:weight:hybrid_powertrain:fuel_cell:CG:x"]
        battery_cg = inputs["data:weight:hybrid_powertrain:battery:CG:x"]

        engine_mass = inputs["data:weight:hybrid_powertrain:engine:mass"]
        fuel_lines_mass = inputs["data:weight:propulsion:fuel_lines:mass"]
        fuel_cells_mass = inputs["data:weight:hybrid_powertrain:fuel_cell:mass"]
        battery_mass = inputs["data:weight:hybrid_powertrain:battery:mass"]

        cg_propulsion = (engine_cg * engine_mass + fuel_lines_cg * fuel_lines_mass +
                         fuel_cell_cg*fuel_cells_mass + battery_cg*battery_mass) / (
            engine_mass + fuel_lines_mass + fuel_cells_mass + battery_mass
        )

        outputs["data:weight:propulsion:CG:x"] = cg_propulsion
