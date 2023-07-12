"""Estimation of fuselage master cross section."""
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
import openmdao.api as om
import fastoad.api as oad

from ..constants import SUBMODEL_FUSELAGE_CROSS_SECTION


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_CROSS_SECTION, "fastga.submodel.geometry.fuselage.cross_section.legacy"
)
class ComputeMasterCrossSection(om.ExplicitComponent):
    """
    Computes the area of fuselage master cross section.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:master_cross_section", units="m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]

        fus_dia = np.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        master_cross_section = np.pi * (fus_dia / 2.0) ** 2.0

        outputs["data:geometry:fuselage:master_cross_section"] = master_cross_section
