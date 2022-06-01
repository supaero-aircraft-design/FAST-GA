"""FAST - Copyright (c) 2016 ONERA ISAE. """
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga.models.geometry.geom_components import (
    ComputeHorizontalTailGeometryFD,
    ComputeHorizontalTailGeometryFL,
    ComputeVerticalTailGeometryFD,
    ComputeVerticalTailGeometryFL,
)
from fastga.models.geometry.geom_components.fuselage.compute_fuselage import (
    ComputeFuselageAlternate,
    ComputeFuselageLegacy,
)

from fastga.models.options import CABIN_SIZING_OPTION

from .constants import (
    SUBMODEL_WING_GEOMETRY,
    SUBMODEL_NACELLE_DIMENSION,
    SUBMODEL_NACELLE_POSITION,
    SUBMODEL_LANDING_GEAR_GEOMETRY,
    SUBMODEL_MFW,
    SUBMODEL_AIRCRAFT_WET_AREA,
    SUBMODEL_PROPELLER_GEOMETRY,
)


@oad.RegisterOpenMDAOSystem("fastga.geometry.alternate", domain=ModelDomain.GEOMETRY)
class GeometryFixedFuselage(om.Group):
    """
    Computes geometric characteristics of the (tube-wing) aircraft:
      - fuselage size is computed from payload requirements
      - wing dimensions are computed from global parameters (area, taper ratio...)
      - tail planes are dimensioned from HQ requirements

    The hypothesis done is a fixed rear_length that fixes fuselage length leading to tail-wing
    distance changes when wing position change. This results in a mainly-linear behaviour on the
    static margin derivative relative to wing position.
    """

    def initialize(self):
        self.options.declare(CABIN_SIZING_OPTION, types=float, default=1.0)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "compute_engine_nacelle_dimension",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_NACELLE_DIMENSION, options=propulsion_option
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_fuselage",
            ComputeFuselageAlternate(
                cabin_sizing=self.options[CABIN_SIZING_OPTION],
                propulsion_id=self.options["propulsion_id"],
            ),
            promotes=["*"],
        )
        self.add_subsystem("compute_vt", ComputeVerticalTailGeometryFL(), promotes=["*"])
        self.add_subsystem("compute_ht", ComputeHorizontalTailGeometryFL(), promotes=["*"])
        self.add_subsystem(
            "compute_wing",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_GEOMETRY),
            promotes=["*"],
        )

        self.add_subsystem(
            "compute_engine_nacelle_position",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_NACELLE_POSITION),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_propeller_geometry",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PROPELLER_GEOMETRY),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_lg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LANDING_GEAR_GEOMETRY),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_tank", oad.RegisterSubmodel.get_submodel(SUBMODEL_MFW), promotes=["*"]
        )
        self.add_subsystem(
            "compute_total_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRCRAFT_WET_AREA),
            promotes=["*"],
        )


@oad.RegisterOpenMDAOSystem("fastga.geometry.legacy", domain=ModelDomain.GEOMETRY)
class GeometryFixedTailDistance(om.Group):
    """
    Computes geometric characteristics of the (tube-wing) aircraft:
      - fuselage size is computed from payload requirements
      - wing dimensions are computed from global parameters (area, taper ratio...)
      - tail planes are dimensioned from HQ requirements

    The hypothesis done is a fixed tail-wing distance leading to fuselage variation (rear_length)
    when wing position changes. This results in a highly non-linear behaviour on the static
    margin derivative relative to wing position.
    """

    def initialize(self):
        self.options.declare(CABIN_SIZING_OPTION, types=float, default=1.0)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "compute_engine_nacelle_dimension",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_NACELLE_DIMENSION, options=propulsion_option
            ),
            promotes=["*"],
        )
        self.add_subsystem("compute_vt", ComputeVerticalTailGeometryFD(), promotes=["*"])
        self.add_subsystem("compute_ht", ComputeHorizontalTailGeometryFD(), promotes=["*"])
        self.add_subsystem(
            "compute_fuselage",
            ComputeFuselageLegacy(
                cabin_sizing=self.options[CABIN_SIZING_OPTION],
                propulsion_id=self.options["propulsion_id"],
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_wing",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_GEOMETRY),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_engine_nacelle",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_NACELLE_POSITION),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_propeller_geometry",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PROPELLER_GEOMETRY),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_lg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LANDING_GEAR_GEOMETRY),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_tank", oad.RegisterSubmodel.get_submodel(SUBMODEL_MFW), promotes=["*"]
        )
        self.add_subsystem(
            "compute_total_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRCRAFT_WET_AREA),
            promotes=["*"],
        )
