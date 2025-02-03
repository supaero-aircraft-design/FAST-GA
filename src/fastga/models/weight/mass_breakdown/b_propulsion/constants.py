"""
Python module for constants of submodels and services naming strings,
applied in propulsion mass calculations.
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

SERVICE_INSTALLED_ENGINE_MASS = "service.weight.mass.propulsion.installed_engine"
SERVICE_FUEL_SYSTEM_MASS = "service.weight.mass.propulsion.fuel_system"
SERVICE_UNUSABLE_FUEL_MASS = "service.weight.mass.propulsion.unusable_fuel"

SUBMODEL_INSTALLED_ENGINE_MASS_LEGACY = (
    "fastga.submodel.weight.mass.propulsion.installed_engine.legacy"
)
SUBMODEL_INSTALLED_ENGINE_MASS_RAYMER = (
    "fastga.submodel.weight.mass.propulsion.installed_engine.raymer"
)

SUBMODEL_FUEL_SYSTEM_MASS_LEGACY = "fastga.submodel.weight.mass.propulsion.fuel_system.legacy"
SUBMODEL_FUEL_SYSTEM_MASS_FLOPS = "fastga.submodel.weight.mass.propulsion.fuel_system.flops"

SUBMODEL_UNUSABLE_FUEL_MASS_LEGACY = "fastga.submodel.weight.mass.propulsion.unusable_fuel.legacy"
