"""Estimation of fuselage wet area FLOPS."""
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

from ..constants import SUBMODEL_FUSELAGE_WET_AREA


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_WET_AREA, "fastga.submodel.geometry.fuselage.wet_area.flops"
)
class ComputeFuselageWetAreaFLOPS(om.ExplicitComponent):
    """
    Fuselage wet area estimation, determined based on Wells, Douglas P., Bryce L. Horvath,
    and Linwood A. McCullers. "The Flight Optimization System Weights Estimation Method." (2017).
    Equation 61.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:wet_area", units="m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]

        # Using the formula from The Flight Optimization System Weights Estimation Method
        fus_dia = np.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        wet_area_fus = np.pi * (fus_length / fus_dia - 1.7) * fus_dia ** 2.0

        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus
