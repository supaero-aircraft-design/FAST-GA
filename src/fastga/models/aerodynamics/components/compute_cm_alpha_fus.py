"""Estimation of fuselage pitching moment."""
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
import fastoad.api as oad

from .figure_digitization import FigureDigitization
from ..constants import SUBMODEL_CM_ALPHA_FUSELAGE


@oad.RegisterSubmodel(
    SUBMODEL_CM_ALPHA_FUSELAGE, "fastga.submodel.aerodynamics.fuselage.pitching_moment.legacy"
)
class ComputeFuselagePitchingMoment(FigureDigitization):
    """
    Estimation of the fuselage pitching moment using the methodology described in section 16.3.8
    of Raymer

    Based on : Raymer, Daniel P. "Aircraft design: a conceptual approach (AIAA Education
    Series)." Reston, Virginia (2012).
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:aerodynamics:fuselage:cm_alpha", units="rad**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        fus_length = inputs["data:geometry:fuselage:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        x0_25 = x_wing - 0.25 * l0_wing - x0_wing + 0.25 * l1_wing
        ratio_x025 = x0_25 / fus_length

        k_fus = self.k_fus(ratio_x025)

        cm_alpha_fus = -k_fus * width_max ** 2 * fus_length / (l0_wing * wing_area) * 180.0 / np.pi

        outputs["data:aerodynamics:fuselage:cm_alpha"] = cm_alpha_fus
