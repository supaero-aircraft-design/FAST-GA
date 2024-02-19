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

from .components import (
    ComputeHTMacFD,
    ComputeHTMacFL,
    ComputeHTSpan,
    ComputeHTRootChord,
    ComputeHTTipChord,
)
from .components import (
    ComputeHTSweep0,
    ComputeHTSweep50,
    ComputeHTSweep100,
)
from .constants import (
    SUBMODEL_HT_WET_AREA,
    SUBMODEL_HT_WET_DISTANCE,
    SUBMODEL_HT_WET_EFFICIENCY,
    SUBMODEL_HT_VOLUME_COEFF,
)


class ComputeHorizontalTailGeometryFD(om.Group):
    """Horizontal tail geometry estimation based on fixed HTP/VTP distance"""

    def setup(self):

        self.add_subsystem("ht_span", ComputeHTSpan(), promotes=["*"])
        self.add_subsystem("ht_root_chord", ComputeHTRootChord(), promotes=["*"])
        self.add_subsystem("ht_tip_chord", ComputeHTTipChord(), promotes=["*"])
        self.add_subsystem("ht_mac", ComputeHTMacFD(), promotes=["*"])
        self.add_subsystem("ht_sweep_0", ComputeHTSweep0(), promotes=["*"])
        self.add_subsystem("ht_sweep_50", ComputeHTSweep50(), promotes=["*"])
        self.add_subsystem("ht_sweep_100", ComputeHTSweep100(), promotes=["*"])
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

        self.add_subsystem("ht_span", ComputeHTSpan(), promotes=["*"])
        self.add_subsystem("ht_root_chord", ComputeHTRootChord(), promotes=["*"])
        self.add_subsystem("ht_tip_chord", ComputeHTTipChord(), promotes=["*"])
        self.add_subsystem("ht_mac", ComputeHTMacFL(), promotes=["*"])
        self.add_subsystem("ht_sweep_0", ComputeHTSweep0(), promotes=["*"])
        self.add_subsystem("ht_sweep_50", ComputeHTSweep50(), promotes=["*"])
        self.add_subsystem("ht_sweep_100", ComputeHTSweep100(), promotes=["*"])
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
