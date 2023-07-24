"""
    Estimation of geometry of horizontal tail.
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

from .constants import (
    SUBMODEL_HT_WET_AREA,
    SUBMODEL_HT_WET_DISTANCE,
    SUBMODEL_HT_WET_EFFICIENCY,
    SUBMODEL_HT_VOLUME_COEFF,
    SUBMODEL_HT_MAC_LENGTH,
    SUBMODEL_HT_MAC_Y,
    SUBMODEL_HT_MAC_X_LOCAL,
    SUBMODEL_HT_MAC_X_WING,
    SUBMODEL_HT_SPAN,
    SUBMODEL_HT_ROOT_CHORD,
    SUBMODEL_HT_TIP_CHORD,
    SUBMODEL_HT_SWEEP_0,
    SUBMODEL_HT_SWEEP_50,
    SUBMODEL_HT_SWEEP_100,
)


class ComputeHorizontalTailGeometryFD(om.Group):
    """Horizontal tail geometry estimation based on fixed HTP/VTP distance"""

    def setup(self):

        self.add_subsystem(
            "ht_span", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SPAN), promotes=["*"]
        )
        self.add_subsystem(
            "ht_root_chord",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_ROOT_CHORD),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_tip_chord", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_TIP_CHORD), promotes=["*"]
        )
        self.add_subsystem(
            "ht_mac_length",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_MAC_LENGTH),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_mac_y", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_MAC_Y), promotes=["*"]
        )
        self.add_subsystem(
            "ht_mac_x_local",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_MAC_X_LOCAL),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_sweep_0", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SWEEP_0), promotes=["*"]
        )
        self.add_subsystem(
            "ht_sweep_50", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SWEEP_50), promotes=["*"]
        )
        self.add_subsystem(
            "ht_sweep_100", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SWEEP_100), promotes=["*"]
        )
        self.add_subsystem(
            "ht_wet_area", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_WET_AREA), promotes=["*"]
        )
        self.add_subsystem(
            "ht_distance",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_WET_DISTANCE),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_eff", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_WET_EFFICIENCY), promotes=["*"]
        )
        self.add_subsystem(
            "ht_volume_coeff",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_VOLUME_COEFF),
            promotes=["*"],
        )


class ComputeHorizontalTailGeometryFL(om.Group):
    """Horizontal tail geometry estimation based on fixed fuselage length"""

    def setup(self):

        self.add_subsystem(
            "ht_span", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SPAN), promotes=["*"]
        )
        self.add_subsystem(
            "ht_root_chord",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_ROOT_CHORD),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_tip_chord", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_TIP_CHORD), promotes=["*"]
        )
        self.add_subsystem(
            "ht_mac_length",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_MAC_LENGTH),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_mac_y", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_MAC_Y), promotes=["*"]
        )
        self.add_subsystem(
            "ht_mac_x_local",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_MAC_X_LOCAL),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_mac_x_wing",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_MAC_X_WING),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_sweep_0", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SWEEP_0), promotes=["*"]
        )
        self.add_subsystem(
            "ht_sweep_50", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SWEEP_50), promotes=["*"]
        )
        self.add_subsystem(
            "ht_sweep_100", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_SWEEP_100), promotes=["*"]
        )
        self.add_subsystem(
            "ht_wet_area", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_WET_AREA), promotes=["*"]
        )
        self.add_subsystem(
            "ht_distance",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_WET_DISTANCE),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_eff", oad.RegisterSubmodel.get_submodel(SUBMODEL_HT_WET_EFFICIENCY), promotes=["*"]
        )
