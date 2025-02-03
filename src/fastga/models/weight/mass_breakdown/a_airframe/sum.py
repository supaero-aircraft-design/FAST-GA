"""
Python module for airframe mass calculation,
part of the Operating Empty Weight (OEW) estimation.
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

import fastoad.api as oad
import openmdao.api as om
from .constants import (
    SERVICE_WING_MASS,
    SERVICE_FUSELAGE_MASS,
    SERVICE_TAIL_MASS,
    SERVICE_FLIGHT_CONTROLS_MASS,
    SERVICE_LANDING_GEAR_MASS,
    SERVICE_PAINT_MASS,
)
from ..constants import SERVICE_AIRFRAME_MASS, SUBMODEL_AIRFRAME_MASS_LEGACY


@oad.RegisterSubmodel(SERVICE_AIRFRAME_MASS, SUBMODEL_AIRFRAME_MASS_LEGACY)
class AirframeWeight(om.Group):
    """Computes mass of airframe."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "wing_weight", oad.RegisterSubmodel.get_submodel(SERVICE_WING_MASS), promotes=["*"]
        )
        self.add_subsystem(
            "fuselage_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_FUSELAGE_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_TAIL_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "flight_controls_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_FLIGHT_CONTROLS_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "landing_gear_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_LANDING_GEAR_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "paint_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_PAINT_MASS),
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
