"""
    Estimation of geometry of fuselage part A - Cabin (Commercial)
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

from openmdao.core.group import Group
from .components import (
    ComputeFuselageWetArea,
    ComputeFuselageGeometryBasic,
    ComputeFuselageGeometryCabinSizingFD,
    ComputeFuselageGeometryCabinSizingFL,
)

from fastga.models.options import CABIN_SIZING_OPTION, FUSELAGE_WET_AREA_OPTION


class ComputeFuselageAlternate(Group):
    def initialize(self):
        self.options.declare(CABIN_SIZING_OPTION, types=float, default=1.0)
        self.options.declare(FUSELAGE_WET_AREA_OPTION, types=float, default=0.0)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        if self.options[CABIN_SIZING_OPTION] == 1.0:
            self.add_subsystem(
                "compute_fuselage_dim",
                ComputeFuselageGeometryCabinSizingFL(propulsion_id=self.options["propulsion_id"]),
                promotes=["*"],
            )
        else:
            self.add_subsystem(
                "compute_fuselage_dim", ComputeFuselageGeometryBasic(), promotes=["*"]
            )
        self.add_subsystem(
            "compute_fus_wet_area",
            ComputeFuselageWetArea(fuselage_wet_area=self.options[FUSELAGE_WET_AREA_OPTION]),
            promotes=["*"],
        )


class ComputeFuselageLegacy(Group):
    def initialize(self):
        self.options.declare(CABIN_SIZING_OPTION, types=float, default=1.0)
        self.options.declare(FUSELAGE_WET_AREA_OPTION, types=float, default=0.0)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        if self.options[CABIN_SIZING_OPTION] == 1.0:
            self.add_subsystem(
                "compute_fuselage_dim",
                ComputeFuselageGeometryCabinSizingFD(propulsion_id=self.options["propulsion_id"]),
                promotes=["*"],
            )
        else:
            self.add_subsystem(
                "compute_fuselage_dim", ComputeFuselageGeometryBasic(), promotes=["*"]
            )
        self.add_subsystem(
            "compute_fus_wet_area",
            ComputeFuselageWetArea(fuselage_wet_area=self.options[FUSELAGE_WET_AREA_OPTION]),
            promotes=["*"],
        )
