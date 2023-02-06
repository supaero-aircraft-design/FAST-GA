"""
Creation of a group to ease the use of the Xfoil Polar ExternalCodeComp in the block analysis
function.
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

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group
from stdatm import Atmosphere

from .xfoil_polar import _DEFAULT_AIRFOIL_FILE, XfoilPolar


class XfoilGroup(Group):
    def initialize(self):

        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("airfoil_file", default=_DEFAULT_AIRFOIL_FILE, types=str)
        self.options.declare("compute_negative_air_angle", default=False, types=bool)

    def setup(self):
        self.add_subsystem(
            "pre_xfoil_group_prep",
            _XfoilGroupPrep(),
            promotes=["data:TLAR:v_approach", "data:Xfoil_pre_processing:reynolds"],
        )
        self.add_subsystem(
            "pre_xfoil_polar",
            XfoilPolar(
                airfoil_folder_path=self.options["airfoil_folder_path"],
                airfoil_file=self.options["airfoil_file"],
                alpha_end=20.0,
                activate_negative_angle=True,
            ),
            promotes=[],
        )

        self.connect("pre_xfoil_group_prep.xfoil_group:mach", "pre_xfoil_polar.xfoil:mach")
        self.connect("pre_xfoil_group_prep.xfoil_group:reynolds", "pre_xfoil_polar.xfoil:reynolds")


class _XfoilGroupPrep(ExplicitComponent):
    """Compute the correct mach number for the Xfoil preprocessing"""

    def setup(self):

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:Xfoil_pre_processing:reynolds", val=np.nan)

        self.add_output("xfoil_group:mach")
        self.add_output("xfoil_group:reynolds")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sos = Atmosphere(0.0).speed_of_sound
        outputs["xfoil_group:mach"] = inputs["data:TLAR:v_approach"] / sos
        outputs["xfoil_group:reynolds"] = inputs["data:Xfoil_pre_processing:reynolds"]
