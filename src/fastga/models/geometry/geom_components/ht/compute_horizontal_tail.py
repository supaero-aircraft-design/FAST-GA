"""
Python module for horizontal tail geometry calculation, part of the geometry component.
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

from .components import ComputeHTMAC, ComputeHTMACFromWing25, ComputeHTVolumeCoefficient
from .constants import (
    SERVICE_HT_CHORD,
    SERVICE_HT_SWEEP,
    SERVICE_HT_WET_AREA,
    SERVICE_HT_DISTANCE,
    SERVICE_HT_EFFICIENCY,
)


# pylint: disable=too-few-public-methods
class ComputeHorizontalTailGeometryFD(om.Group):
    """Horizontal tail geometry estimation based on fixed HTP/VTP distance."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "ht_chord", oad.RegisterSubmodel.get_submodel(SERVICE_HT_CHORD), promotes=["*"]
        )
        self.add_subsystem("ht_mac", ComputeHTMAC(), promotes=["*"])
        self.add_subsystem(
            "ht_sweep", oad.RegisterSubmodel.get_submodel(SERVICE_HT_SWEEP), promotes=["*"]
        )
        self.add_subsystem(
            "ht_wet_area", oad.RegisterSubmodel.get_submodel(SERVICE_HT_WET_AREA), promotes=["*"]
        )
        self.add_subsystem(
            "ht_distance",
            oad.RegisterSubmodel.get_submodel(SERVICE_HT_DISTANCE),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_eff", oad.RegisterSubmodel.get_submodel(SERVICE_HT_EFFICIENCY), promotes=["*"]
        )
        self.add_subsystem("ht_volume_coeff", ComputeHTVolumeCoefficient(), promotes=["*"])


# pylint: disable=too-few-public-methods
class ComputeHorizontalTailGeometryFL(om.Group):
    """Horizontal tail geometry estimation based on fixed rear fuselage length."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "ht_chord", oad.RegisterSubmodel.get_submodel(SERVICE_HT_CHORD), promotes=["*"]
        )
        self.add_subsystem("ht_mac", ComputeHTMAC(), promotes=["*"])
        self.add_subsystem("ht_mac_from_wing_25", ComputeHTMACFromWing25(), promotes=["*"])
        self.add_subsystem(
            "ht_sweep", oad.RegisterSubmodel.get_submodel(SERVICE_HT_SWEEP), promotes=["*"]
        )
        self.add_subsystem(
            "ht_wet_area", oad.RegisterSubmodel.get_submodel(SERVICE_HT_WET_AREA), promotes=["*"]
        )
        self.add_subsystem(
            "ht_distance",
            oad.RegisterSubmodel.get_submodel(SERVICE_HT_DISTANCE),
            promotes=["*"],
        )
        self.add_subsystem(
            "ht_eff", oad.RegisterSubmodel.get_submodel(SERVICE_HT_EFFICIENCY), promotes=["*"]
        )
