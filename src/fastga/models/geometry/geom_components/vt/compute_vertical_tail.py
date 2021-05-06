"""
    Estimation of geometry of vertical tail
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

from .components import ComputeVTChords, ComputeVTmacFD, ComputeVTmacFL, ComputeVTSweep, ComputeVTWetArea


class ComputeVerticalTailGeometryFD(om.Group):
    """ Vertical tail geometry estimation based on fixed HTP/VTP distance """

    def setup(self):
        
        self.add_subsystem("vt_chords", ComputeVTChords(), promotes=["*"])
        self.add_subsystem("vt_mac", ComputeVTmacFD(), promotes=["*"])
        self.add_subsystem("vt_sweep", ComputeVTSweep(), promotes=["*"])
        self.add_subsystem("vt_wet_area", ComputeVTWetArea(), promotes=["*"])


class ComputeVerticalTailGeometryFL(om.Group):
    """ Vertical tail geometry estimation based on fixed fuselage length """

    def setup(self):
        self.add_subsystem("vt_chords", ComputeVTChords(), promotes=["*"])
        self.add_subsystem("vt_mac", ComputeVTmacFL(), promotes=["*"])
        self.add_subsystem("vt_sweep", ComputeVTSweep(), promotes=["*"])
        self.add_subsystem("vt_wet_area", ComputeVTWetArea(), promotes=["*"])
