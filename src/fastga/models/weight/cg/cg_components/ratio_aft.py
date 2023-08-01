"""
    Estimation of center of gravity ratio with aft.
"""
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

from ..cg_components.constants import (
    SUBMODEL_WING_CG,
    SUBMODEL_FUSELAGE_CG,
    SUBMODEL_HORIZONTAL_TAIL_CG,
    SUBMODEL_VERTICAL_TAIL_CG,
    SUBMODEL_FLIGHT_CONTROLS_CG,
    SUBMODEL_MAIN_LANDING_GEAR_CG,
    SUBMODEL_FRONT_LANDING_GEAR_CG,
    SUBMODEL_PROPULSION_CG,
    SUBMODEL_ELECTRIC_POWER_SYSTEMS_CG,
    SUBMODEL_HYDRAULIC_POWER_SYSTEMS_CG,
    SUBMODEL_LIFE_SUPPORT_SYSTEMS_CG,
    SUBMODEL_NAVIGATION_SYSTEMS_CG,
    SUBMODEL_RECORDING_SYSTEMS_CG,
    SUBMODEL_SEATS_CG,
    SUBMODEL_AIRCRAFT_X_CG,
    SUBMODEL_AIRCRAFT_X_CG_RATIO,
    SUBMODEL_AIRCRAFT_Z_CG,
    SUBMODEL_AIRCRAFT_EMPTY_MASS,
    SUBMODEL_ENGINE_Z_CG,
)

from ..cg_components.aircraft_empty_cg_ratio import ComputeCGRatio


@oad.RegisterSubmodel(
    SUBMODEL_AIRCRAFT_X_CG_RATIO, "fastga.submodel.weight.cg.aircraft_empty.x_ratio.legacy"
)
class ComputeCGRatioAircraftEmpty(om.Group):
    def setup(self):
        self.add_subsystem(
            "wing_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_WING_CG), promotes=["*"]
        )
        self.add_subsystem(
            "fuselage_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_FUSELAGE_CG), promotes=["*"]
        )
        self.add_subsystem(
            "horizontal_tail_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HORIZONTAL_TAIL_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "vertical_tail_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VERTICAL_TAIL_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "flight_control_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FLIGHT_CONTROLS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "main_landing_gear_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_MAIN_LANDING_GEAR_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "front_landing_gear_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_FRONT_LANDING_GEAR_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "propulsion_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_PROPULSION_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "electric_power_systems_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_ELECTRIC_POWER_SYSTEMS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "hydraulic_power_systems_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HYDRAULIC_POWER_SYSTEMS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "life_support_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_LIFE_SUPPORT_SYSTEMS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "navigation_systems_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_NAVIGATION_SYSTEMS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "recording_systems_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_RECORDING_SYSTEMS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "passenger_seats_cg",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_SEATS_CG),
            promotes=["*"],
        )
        self.add_subsystem(
            "total_mass_empty",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRCRAFT_EMPTY_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "x_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRCRAFT_X_CG), promotes=["*"]
        )
        self.add_subsystem(
            "z_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_AIRCRAFT_Z_CG), promotes=["*"]
        )
        self.add_subsystem(
            "engine_z_cg", oad.RegisterSubmodel.get_submodel(SUBMODEL_ENGINE_Z_CG), promotes=["*"]
        )
        self.add_subsystem("cg_ratio", ComputeCGRatio(), promotes=["*"])
