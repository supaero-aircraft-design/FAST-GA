"""
    Estimation of geometry of horizontal tail
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

from .components import ComputeHTChord, ComputeHTDistance, ComputeHTmacFD, ComputeHTmacFL, ComputeHTSweep, \
    ComputeHTWetArea, ComputeHTEfficiency


class ComputeHorizontalTailGeometryFD(om.Group):
    """ Horizontal tail geometry estimation based on fixed HTP/VTP distance """

    def setup(self):
        self.add_subsystem("ht_chord", ComputeHTChord(), promotes=["*"])
        self.add_subsystem("ht_mac", ComputeHTmacFD(), promotes=["*"])
        self.add_subsystem("ht_sweep", ComputeHTSweep(), promotes=["*"])
        self.add_subsystem("ht_wet_area", ComputeHTWetArea(), promotes=["*"])
        self.add_subsystem("ht_distance", ComputeHTDistance(), promotes=["*"])
        self.add_subsystem("ht_eff", ComputeHTEfficiency(), promotes=["*"])


class ComputeHorizontalTailGeometryFL(om.Group):
    """ Horizontal tail geometry estimation based on fixed fuselage length """

    def setup(self):
        self.add_subsystem("ht_chord", ComputeHTChord(), promotes=["*"])
        self.add_subsystem("ht_mac", ComputeHTmacFL(), promotes=["*"])
        self.add_subsystem("ht_sweep", ComputeHTSweep(), promotes=["*"])
        self.add_subsystem("ht_wet_area", ComputeHTWetArea(), promotes=["*"])
        self.add_subsystem("ht_distance", ComputeHTDistance(), promotes=["*"])
