""" Module that computes the powertrain in a hybrid propulsion model (FC-B configuration). """

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
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem, RegisterSubmodel
from fastoad.module_management.constants import ModelDomain

from .components import (
    ComputeBatteries,
    ComputeBoP,
    ComputeCompressor,
    ComputeElectricMotor,
    ComputeFuelCells,
    ComputeH2Storage,
    ComputeHex,
    ComputeInverter,
    ComputeIntakes
)

@RegisterOpenMDAOSystem("fastga.hybrid_powertrain.legacy", domain=ModelDomain.GEOMETRY)
class ComputeHybridPowertrain(om.Group):

    def initialize(self):
        self.options.declare('H2_storage_model', types=str, default='legacy')

    def setup(self):
        self.add_subsystem("compute_fuel_cells", ComputeFuelCells(), promotes=["*"])
        self.add_subsystem("compute_compressor", ComputeCompressor(), promotes=["*"])
        self.add_subsystem("compute_bop", ComputeBoP(), promotes=["*"])
        self.add_subsystem("compute_hex", ComputeHex(), promotes=["*"])
        self.add_subsystem("compute_intakes", ComputeIntakes(), promotes=["*"])
        self.add_subsystem("compute_batteries", ComputeBatteries(), promotes=["*"])
        self.add_subsystem("compute_inverter", ComputeInverter(), promotes=["*"])
        self.add_subsystem("compute_electric_motor", ComputeElectricMotor(), promotes=["*"])
        self.add_subsystem("compute_h2_storage", ComputeH2Storage(), promotes=["*"])

        self.approx_totals()
