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

# from .compute_bending_moment import ComputeBendingMoment
from .compute_torsion_moment import ComputeTorsionMoment
from .compute_fuselage_shell import ComputeFuselageShell
from .compute_bending_moment_bis import ComputeBendingMomentBis
from .compute_fuselage_additional_mass import ComputeFuselageAdditionalMass


class LoadsFuselage(om.Group):

    def setup(self):
        # self.add_subsystem("fuselage_bending_moment", ComputeBendingMoment(), promotes=["*"])
        self.add_subsystem("fuselage_torsion_moment", ComputeTorsionMoment(), promotes=["*"])
        self.add_subsystem("fuselage_shell", ComputeFuselageShell(), promotes=["*"])
        self.add_subsystem("fuselage_additional_mass", ComputeFuselageAdditionalMass(), promotes=["*"])
        self.add_subsystem("fuselage_bending_moment", ComputeBendingMomentBis(), promotes=["*"])
