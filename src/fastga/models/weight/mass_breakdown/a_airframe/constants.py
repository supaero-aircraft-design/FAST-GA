"""
Python module for constants of submodels and services naming strings,
applied in airframe mass calculations.
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

SERVICE_WING_MASS = "service.weight.mass.airframe.wing"
SERVICE_FUSELAGE_MASS = "service.weight.mass.airframe.fuselage"
SERVICE_TAIL_MASS = "service.weight.mass.airframe.tail"
SERVICE_HTP_MASS = "service.weight.mass.airframe.tail.htp"
SERVICE_VTP_MASS = "service.weight.mass.airframe.tail.vtp"
SERVICE_FLIGHT_CONTROLS_MASS = "service.weight.mass.airframe.flight_controls"
SERVICE_LANDING_GEAR_MASS = "service.weight.mass.airframe.landing_gear"
SERVICE_PAINT_MASS = "service.weight.mass.airframe.paint"

SUBMODEL_TAIL_MASS_LEGACY = "fastga.submodel.weight.mass.airframe.tail.legacy"
SUBMODEL_TAIL_MASS_GD = "fastga.submodel.weight.mass.airframe.tail.gd"
SUBMODEL_TAIL_MASS_TORENBEEKGD = "fastga.submodel.weight.mass.airframe.tail.torenbeek_gd"

SUBMODEL_HTP_MASS_LEGACY = "fastga.submodel.weight.mass.airframe.tail.htp.legacy"
SUBMODEL_HTP_MASS_GD = "fastga.submodel.weight.mass.airframe.tail.htp.gd"
SUBMODEL_HTP_MASS_TORENBEEK = "fastga.submodel.weight.mass.airframe.tail.htp.torenbeek"

SUBMODEL_VTP_MASS_LEGACY = "fastga.submodel.weight.mass.airframe.tail.vtp.legacy"
SUBMODEL_VTP_MASS_GD = "fastga.submodel.weight.mass.airframe.tail.vtp.gd"

SUBMODEL_WING_MASS_LEGACY = "fastga.submodel.weight.mass.airframe.wing.legacy"
SUBMODEL_WING_MASS_ANALYTICAL = "fastga.submodel.weight.mass.airframe.wing.analytical"

SUBMODEL_FUSELAGE_MASS_LEGACY = "fastga.submodel.weight.mass.airframe.fuselage.legacy"
SUBMODEL_FUSELAGE_MASS_RAYMER = "fastga.submodel.weight.mass.airframe.fuselage.raymer"
SUBMODEL_FUSELAGE_MASS_ROSKAM = "fastga.submodel.weight.mass.airframe.fuselage.roskam"
SUBMODEL_FUSELAGE_MASS_ANALYTICAL = "fastga.submodel.weight.mass.airframe.fuselage.analytical"

SUBMODEL_FLIGHT_CONTROLS_MASS_LEGACY = "fastga.submodel.weight.mass.airframe.flight_controls.legacy"
SUBMODEL_FLIGHT_CONTROLS_MASS_FLOPS = "fastga.submodel.weight.mass.airframe.flight_controls.flops"

SUBMODEL_LANDING_GEAR_MASS_LEGACY = "fastga.submodel.weight.mass.airframe.landing_gear.legacy"

SUBMODEL_PAINT_MASS_NO_PAINT = "fastga.submodel.weight.mass.airframe.paint.no_paint"
SUBMODEL_PAINT_MASS_BY_WET_AREA = "fastga.submodel.weight.mass.airframe.paint.by_wet_area"
