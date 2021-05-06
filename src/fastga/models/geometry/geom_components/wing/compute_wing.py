"""
    Estimation of wing geometry
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

from .components import ComputeMFW
from .components import ComputeWingB50
from .components import ComputeWingL1AndL4
from .components import ComputeWingL2AndL3
from .components import ComputeWingMAC
from .components import ComputeWingSweep
from .components import ComputeWingToc
from .components import ComputeWingWetArea
from .components import ComputeWingX
from .components import ComputeWingY

from openmdao.api import Group


class ComputeWingGeometry(Group):
    # TODO: Document equations. Cite sources
    """ Wing geometry estimation """

    def setup(self):
        self.add_subsystem("wing_toc", ComputeWingToc(), promotes=["*"])
        self.add_subsystem("wing_y", ComputeWingY(), promotes=["*"])
        self.add_subsystem("wing_l1l4", ComputeWingL1AndL4(), promotes=["*"])
        self.add_subsystem("wing_l2l3", ComputeWingL2AndL3(), promotes=["*"])
        self.add_subsystem("wing_x", ComputeWingX(), promotes=["*"])
        self.add_subsystem("wing_b50", ComputeWingB50(), promotes=["*"])
        self.add_subsystem("wing_mac", ComputeWingMAC(), promotes=["*"])
        self.add_subsystem("wing_sweep", ComputeWingSweep(), promotes=["*"])
        self.add_subsystem("wing_wet_area", ComputeWingWetArea(), promotes=["*"])
        self.add_subsystem("mfw", ComputeMFW(), promotes=["*"])
