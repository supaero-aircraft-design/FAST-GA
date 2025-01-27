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

import fastoad.api as oad
import openmdao.api as om
import logging
from .constants import (
    SUBMODEL_WING_MASS,
    SUBMODEL_FUSELAGE_MASS,
    SUBMODEL_TAIL_MASS,
    SUBMODEL_HTP_MASS,
    SUBMODEL_VTP_MASS,
    SUBMODEL_FLIGHT_CONTROLS_MASS,
    SUBMODEL_LANDING_GEAR_MASS,
    SUBMODEL_PAINT_MASS,
    TAIL_WEIGHT_LEGACY,
    TAIL_WEIGHT_GD,
    TAIL_WEIGHT_TORENBEEK_GD,
)
from ..constants import SUBMODEL_AIRFRAME_MASS

_LOGGER = logging.getLogger(__name__)

# Set up as default calculation for both HTP and VTP
oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] = TAIL_WEIGHT_LEGACY


@oad.RegisterSubmodel(SUBMODEL_AIRFRAME_MASS, "fastga.submodel.weight.mass.airframe.legacy")
class AirframeWeight(om.Group):
    """Computes mass of airframe."""

    def setup(self):
        _empennage_submodel_check()
        self.add_subsystem(
            "wing_weight", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_MASS), promotes=["*"]
        )
        self.add_subsystem(
            "fuselage_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "flight_controls_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FLIGHT_CONTROLS_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "landing_gear_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LANDING_GEAR_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "paint_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PAINT_MASS),
            promotes=["*"],
        )

        weight_sum = om.AddSubtractComp()
        weight_sum.add_equation(
            "data:weight:airframe:mass",
            [
                "data:weight:airframe:wing:mass",
                "data:weight:airframe:fuselage:mass",
                "data:weight:airframe:horizontal_tail:mass",
                "data:weight:airframe:vertical_tail:mass",
                "data:weight:airframe:flight_controls:mass",
                "data:weight:airframe:landing_gear:main:mass",
                "data:weight:airframe:landing_gear:front:mass",
                "data:weight:airframe:paint:mass",
            ],
            units="kg",
            desc="Mass of the airframe",
        )

        self.add_subsystem("airframe_weight_sum", weight_sum, promotes=["*"])


def _empennage_submodel_check():
    """Check on tail weight mass submodel definition."""

    set_htp_submodel = oad.RegisterSubmodel.active_models.get(SUBMODEL_HTP_MASS)
    set_vtp_submodel = oad.RegisterSubmodel.active_models.get(SUBMODEL_VTP_MASS)

    if oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] == TAIL_WEIGHT_LEGACY:
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
            "fastga.submodel.weight.mass.airframe.htp.legacy"
        )
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
            "fastga.submodel.weight.mass.airframe.vtp.legacy"
        )

    elif oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] == TAIL_WEIGHT_GD:
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
            "fastga.submodel.weight.mass.airframe.htp.gd"
        )
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
            "fastga.submodel.weight.mass.airframe.vtp.gd"
        )

    elif oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] == TAIL_WEIGHT_TORENBEEK_GD:
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
            "fastga.submodel.weight.mass.airframe.htp.torenbeek"
        )
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
            "fastga.submodel.weight.mass.airframe.vtp.gd"
        )

    if set_htp_submodel:
        oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = set_htp_submodel

    if set_vtp_submodel:
        oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = set_vtp_submodel
