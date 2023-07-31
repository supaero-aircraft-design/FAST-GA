"""Computation of the airframe mass."""
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
    SUBMODEL_WING_MASS,
    SUBMODEL_FUSELAGE_MASS,
    SUBMODEL_HORIZONTAL_TAIL_MASS,
    SUBMODEL_VERTICAL_TAIL_MASS,
    SUBMODEL_FLIGHT_CONTROLS_MASS,
    SUBMODEL_FRONT_LANDING_GEAR_MASS,
    SUBMODEL_MAIN_LANDING_GEAR_MASS,
    SUBMODEL_PAINT_MASS,
)

from ..constants import SUBMODEL_AIRFRAME_MASS
from ..a_airframe import ComputeAirframeMass


@oad.RegisterSubmodel(SUBMODEL_AIRFRAME_MASS, "fastga.submodel.weight.mass.airframe.legacy")
class AirframeWeight(om.Group):
    """Computes mass of airframe."""

    def setup(self):
        self.add_subsystem(
            "wing_weight", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_MASS), promotes=["*"]
        )
        self.add_subsystem(
            "fuselage_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "horizontal_tail_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HORIZONTAL_TAIL_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "vertical_tail_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VERTICAL_TAIL_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "flight_controls_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FLIGHT_CONTROLS_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "front_landing_gear_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FRONT_LANDING_GEAR_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "main_landing_gear_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_MAIN_LANDING_GEAR_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "paint_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PAINT_MASS),
            promotes=["*"],
        )
        self.add_subsystem("airframe_weight_sum", ComputeAirframeMass(), promotes=["*"])
