"""
    Estimation of geometry of vertical tail
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

from .components import ComputeVTMacFD, ComputeVTMacFL
from .constants import (
    SUBMODEL_VT_ROOT_CHORD,
    SUBMODEL_VT_TIP_CHORD,
    SUBMODEL_VT_SPAN,
    SUBMODEL_VT_SWEEP_0,
    SUBMODEL_VT_SWEEP_50,
    SUBMODEL_VT_SWEEP_100,
    SUBMODEL_VT_WET_AREA,
    SUBMODEL_VT_POSITION_FD,
    SUBMODEL_VT_POSITION_FL,
)


class ComputeVerticalTailGeometryFD(om.Group):
    """Vertical tail geometry estimation based on fixed HTP/VTP distance"""

    def setup(self):

        self.add_subsystem(
            "vt_root_chord", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_ROOT_CHORD), promotes=["*"]
        )
        self.add_subsystem(
            "vt_tip_chord", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_TIP_CHORD), promotes=["*"]
        )
        self.add_subsystem(
            "vt_span", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SPAN), promotes=["*"]
        )
        self.add_subsystem("vt_mac", ComputeVTMacFD(), promotes=["*"])
        self.add_subsystem(
            "vt_position",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_POSITION_FD),
            promotes=["*"],
        )
        self.add_subsystem(
            "vt_sweep_0", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SWEEP_0), promotes=["*"]
        )
        self.add_subsystem(
            "vt_sweep_50", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SWEEP_50), promotes=["*"]
        )
        self.add_subsystem(
            "vt_sweep_100", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SWEEP_100), promotes=["*"]
        )
        self.add_subsystem(
            "vt_wet_area", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_WET_AREA), promotes=["*"]
        )


class ComputeVerticalTailGeometryFL(om.Group):
    """Vertical tail geometry estimation based on fixed fuselage length"""

    def setup(self):

        self.add_subsystem(
            "vt_root_chord", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_ROOT_CHORD), promotes=["*"]
        )
        self.add_subsystem(
            "vt_tip_chord", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_TIP_CHORD), promotes=["*"]
        )
        self.add_subsystem(
            "vt_span", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SPAN), promotes=["*"]
        )
        self.add_subsystem("vt_mac", ComputeVTMacFL(), promotes=["*"])
        self.add_subsystem(
            "vt_position",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_POSITION_FL),
            promotes=["*"],
        )
        self.add_subsystem(
            "vt_sweep_0", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SWEEP_0), promotes=["*"]
        )
        self.add_subsystem(
            "vt_sweep_50", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SWEEP_50), promotes=["*"]
        )
        self.add_subsystem(
            "vt_sweep_100", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_SWEEP_100), promotes=["*"]
        )
        self.add_subsystem(
            "vt_wet_area", oad.RegisterSubmodel.get_submodel(SUBMODEL_VT_WET_AREA), promotes=["*"]
        )
