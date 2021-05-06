"""
    FAST - Copyright (c) 2016 ONERA ISAE
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

import openmdao.api as om

from .geom_components import ComputeTotalArea, ComputeFuselageGeometryBasic, ComputeFuselageGeometryCabinSizingFD, \
    ComputeHorizontalTailGeometryFD, ComputeNacelleGeometry, ComputeVerticalTailGeometryFD, ComputeWingGeometry
from models.options import CABIN_SIZING_OPTION

from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain


@RegisterOpenMDAOSystem("fastga.geometry.legacy", domain=ModelDomain.GEOMETRY)
class GeometryFixedTailDistance(om.Group):
    """
    Computes geometric characteristics of the (tube-wing) aircraft:
      - fuselage size is computed from payload requirements
      - wing dimensions are computed from global parameters (area, taper ratio...)
      - tail planes are dimensioned from HQ requirements

    The hypothesis done is a fixed tail-wing distance leading to fuselage variation (rear_length) when wing position
    changes. This results in a highly non-linear behaviour on the static margin derivative relative to wing position.
    """

    def initialize(self):
        self.options.declare(CABIN_SIZING_OPTION, types=float, default=1.0)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        
        self.add_subsystem("compute_vt", ComputeVerticalTailGeometryFD(), promotes=["*"])
        self.add_subsystem("compute_ht", ComputeHorizontalTailGeometryFD(), promotes=["*"])
        if self.options[CABIN_SIZING_OPTION] == 1.0:
            self.add_subsystem(
                "compute_fuselage", ComputeFuselageGeometryCabinSizingFD(propulsion_id=self.options["propulsion_id"]),
                promotes=["*"]
            )
        else:
            self.add_subsystem("compute_fuselage", ComputeFuselageGeometryBasic(), promotes=["*"])
        self.add_subsystem("compute_wing", ComputeWingGeometry(), promotes=["*"])
        self.add_subsystem(
            "compute_engine_nacelle", ComputeNacelleGeometry(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"]
        )
        self.add_subsystem("compute_total_area", ComputeTotalArea(), promotes=["*"])
