"""
Python module for vertical tail geometry calculation, part of the geometry component.
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

import openmdao.api as om
import fastoad.api as oad

from .components import ComputeVTMAC, ComputeVTMACDistanceXLocal
from .constants import (
    SERVICE_VT_CHORD,
    SERVICE_VT_SWEEP,
    SERVICE_VT_WET_AREA,
    SERVICE_VT_DISTANCE_FL,
    SERVICE_VT_DISTANCE_FD,
)


# pylint: disable=too-few-public-methods
class ComputeVerticalTailGeometryFD(om.Group):
    """Vertical tail geometry estimation based on fixed HTP/VTP distance."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "vt_chords", oad.RegisterSubmodel.get_submodel(SERVICE_VT_CHORD), promotes=["*"]
        )
        self.add_subsystem("vt_mac", ComputeVTMAC(), promotes=["*"])
        self.add_subsystem("vt_mac_x_local_distance", ComputeVTMACDistanceXLocal(), promotes=["*"])
        self.add_subsystem(
            "vt_distance",
            oad.RegisterSubmodel.get_submodel(SERVICE_VT_DISTANCE_FD),
            promotes=["*"],
        )
        self.add_subsystem(
            "vt_sweep", oad.RegisterSubmodel.get_submodel(SERVICE_VT_SWEEP), promotes=["*"]
        )
        self.add_subsystem(
            "vt_wet_area", oad.RegisterSubmodel.get_submodel(SERVICE_VT_WET_AREA), promotes=["*"]
        )


# pylint: disable=too-few-public-methods
class ComputeVerticalTailGeometryFL(om.Group):
    """Vertical tail geometry estimation based on fixed rear fuselage length"""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "vt_chords", oad.RegisterSubmodel.get_submodel(SERVICE_VT_CHORD), promotes=["*"]
        )
        self.add_subsystem("vt_mac", ComputeVTMAC(), promotes=["*"])
        self.add_subsystem("vt_mac_x_local_distance", ComputeVTMACDistanceXLocal(), promotes=["*"])
        self.add_subsystem(
            "vt_distance",
            oad.RegisterSubmodel.get_submodel(SERVICE_VT_DISTANCE_FL),
            promotes=["*"],
        )
        self.add_subsystem(
            "vt_sweep", oad.RegisterSubmodel.get_submodel(SERVICE_VT_SWEEP), promotes=["*"]
        )
        self.add_subsystem(
            "vt_wet_area", oad.RegisterSubmodel.get_submodel(SERVICE_VT_WET_AREA), promotes=["*"]
        )
