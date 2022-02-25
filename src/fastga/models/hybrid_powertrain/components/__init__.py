""" Computes the hybrid powertrain dimensions in a Fuel Cell - Battery configuration. """

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

from .compute_batteries import ComputeBatteries
from .compute_bop import ComputeBoP
from .compute_compressor import ComputeCompressor
from .compute_electric_motor import ComputeElectricMotor
from .compute_fuel_cells import ComputeFuelCells
from .compute_h2_storage import ComputeH2Storage
from .compute_hex import ComputeHex
from .compute_inverter import ComputeInverter
from .compute_intakes import ComputeIntakes
