"""Estimation of global center of gravity."""
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

from openmdao.api import Group

import fastoad.api as oad

from .max_cg_ratio import ComputeMaxMinCGRatio

from .constants import (
    SUBMODEL_AIRCRAFT_X_CG_RATIO,
    SUBMODEL_LOADCASE_GROUND_X,
    SUBMODEL_LOADCASE_FLIGHT_X,
    SUBMODEL_AIRCRAFT_CG_EXTREME,
)


@oad.RegisterSubmodel(SUBMODEL_AIRCRAFT_CG_EXTREME, "fastga.submodel.weight.cg.aircraft.x.legacy")
class ComputeGlobalCG(Group):
    # TODO: Document equations. Cite sources
    """Global center of gravity estimation."""

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "cg_ratio_aft",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRCRAFT_X_CG_RATIO),
            promotes=["*"],
        )
        self.add_subsystem(
            "cg_ratio_lc_ground",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LOADCASE_GROUND_X),
            promotes=["*"],
        )
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "cg_ratio_lc_flight",
            oad.RegisterSubmodel.get_submodel(
                SUBMODEL_LOADCASE_FLIGHT_X, options=propulsion_option
            ),
            promotes=["*"],
        )
        self.add_subsystem("cg_ratio_extrema", ComputeMaxMinCGRatio(), promotes=["*"])
