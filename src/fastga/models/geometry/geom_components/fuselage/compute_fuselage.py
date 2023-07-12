"""Estimation of geometry of fuselage part A - Cabin (Commercial)."""
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

from .constants import (
    SUBMODEL_FUSELAGE_WET_AREA,
    SUBMODEL_FUSELAGE_DEPTH,
    SUBMODEL_FUSELAGE_VOLUME,
    SUBMODEL_FUSELAGE_CROSS_SECTION,
    SUBMODEL_FUSELAGE_NPAX,
    SUBMODEL_FUSELAGE_PAX_LENGTH,
    SUBMODEL_FUSELAGE_MAX_WIDTH,
    SUBMODEL_FUSELAGE_MAX_HEIGHT,
    SUBMODEL_FUSELAGE_LUGGAGE_LENGTH,
    SUBMODEL_FUSELAGE_CABIN_LENGTH,
    SUBMODEL_FUSELAGE_NOSE_LENGTH_FL,
    SUBMODEL_FUSELAGE_NOSE_LENGTH_FD,
    SUBMODEL_FUSELAGE_LENGTH_FL,
    SUBMODEL_FUSELAGE_LENGTH_FD,
    SUBMODEL_FUSELAGE_PLANE_LENGTH,
    SUBMODEL_FUSELAGE_REAR_LENGTH,
    SUBMODEL_FUSELAGE_DIMENSIONS_BASIC,
)

from fastga.models.options import CABIN_SIZING_OPTION


class ComputeFuselageAlternate(om.Group):
    def initialize(self):
        self.options.declare(CABIN_SIZING_OPTION, types=float, default=1.0)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        if self.options[CABIN_SIZING_OPTION] == 1.0:
            self.add_subsystem(
                "compute_fuselage_npax",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_NPAX),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_pax_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_PAX_LENGTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_max_width",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_MAX_WIDTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_max_height",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_MAX_HEIGHT),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_luggage_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_LUGGAGE_LENGTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_cabin_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_CABIN_LENGTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_nose_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_NOSE_LENGTH_FL),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_LENGTH_FL),
                promotes=["*"],
            )
        else:
            self.add_subsystem(
                "compute_fuselage_dim", 
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_DIMENSIONS_BASIC), 
                promotes=["*"],
            )
        self.add_subsystem(
            "compute_fus_wet_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_WET_AREA),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_fuselage_master_cross_section_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_CROSS_SECTION),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_avg_fuselage_depth",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_DEPTH),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_fuselage_volume",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_VOLUME),
            promotes=["*"],
        )


class ComputeFuselageLegacy(om.Group):
    def initialize(self):
        self.options.declare(CABIN_SIZING_OPTION, types=float, default=1.0)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        if self.options[CABIN_SIZING_OPTION] == 1.0:
            self.add_subsystem(
                "compute_fuselage_npax",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_NPAX),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_pax_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_PAX_LENGTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_max_width",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_MAX_WIDTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_max_height",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_MAX_HEIGHT),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_luggage_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_LUGGAGE_LENGTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_cabin_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_CABIN_LENGTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_nose_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_NOSE_LENGTH_FD),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_LENGTH_FD),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_plane_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_PLANE_LENGTH),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_fuselage_rear_length",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_REAR_LENGTH),
                promotes=["*"],
            )
        else:
            self.add_subsystem(
                "compute_fuselage_dim", 
                oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_DIMENSIONS_BASIC), 
                promotes=["*"],
            )
        self.add_subsystem(
            "compute_fus_wet_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_WET_AREA),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_fuselage_master_cross_section_area",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_CROSS_SECTION),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_avg_fuselage_depth",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_DEPTH),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_fuselage_volume",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_VOLUME),
            promotes=["*"],
        )
