"""
Python module for constants of submodels and services naming strings,
applied in systems mass calculations.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

SERVICE_POWER_SYSTEM_MASS = "service.weight.mass.system.power_system"
SERVICE_LIFE_SUPPORT_SYSTEM_MASS = "service.weight.mass.system.life_support_system"
SERVICE_AVIONICS_SYSTEM_MASS = "service.weight.mass.system.avionics_system"
SERVICE_RECORDING_SYSTEM_MASS = "service.weight.mass.system.recording_system"

SUBMODEL_POWER_SYSTEM_MASS_LEGACY = "fastga.submodel.weight.mass.system.power_system.legacy"

SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS_LEGACY = (
    "fastga.submodel.weight.mass.system.life_support_system.legacy"
)
SUBMODEL_LIFE_SUPPORT_SYSTEM_MASS_FLOPS = (
    "fastga.submodel.weight.mass.system.life_support_system.flops"
)

SUBMODEL_AVIONICS_SYSTEM_MASS_LEGACY = "fastga.submodel.weight.mass.system.avionics_systems.legacy"
SUBMODEL_AVIONICS_SYSTEM_MASS_FROM_UNINSTALLED = (
    "fastga.submodel.weight.mass.system.avionics_systems.from_uninstalled"
)

SUBMODEL_RECORDING_SYSTEM_MASS_MINIMUM = (
    "fastga.submodel.weight.mass.system.recording_systems.minimum"
)
