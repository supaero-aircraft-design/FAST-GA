"""Estimation of yawing moment due to side-slip."""

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
from openmdao.core.explicitcomponent import ExplicitComponent

from ...constants import SUBMODEL_CN_BETA_FUSELAGE


@oad.RegisterSubmodel(
    SUBMODEL_CN_BETA_FUSELAGE, "fastga.submodel.aerodynamics.fuselage.yawing_moment_beta.legacy"
)
class ComputeCnBetaFuselage(ExplicitComponent):
    """
    Yawing moment due to side-slip estimation.

    Based on : Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012. Sixth Edition, equation 16.50.
    """

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:volume", val=np.nan, units="m**3")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output("data:aerodynamics:fuselage:Cn_beta", units="rad**-1")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        volume_fus = inputs["data:geometry:fuselage:volume"]
        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]

        l_f = np.sqrt(width_max * height_max)
        # estimation of fuselage volume
        # equation from raymer book eqn. 16.47
        cn_beta = -1.3 * volume_fus / wing_area / span * (l_f / width_max)

        outputs["data:aerodynamics:fuselage:Cn_beta"] = cn_beta

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        volume_fus = inputs["data:geometry:fuselage:volume"]
        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]

        l_f = np.sqrt(width_max * height_max)

        partials["data:aerodynamics:fuselage:Cn_beta", "data:geometry:fuselage:maximum_width"] = (
            1.0 / 2.0 * 1.3 * volume_fus / wing_area / span * (l_f / width_max ** 2)
        )
        partials["data:aerodynamics:fuselage:Cn_beta", "data:geometry:fuselage:maximum_height"] = (
            -1.0 / 2.0 * 1.3 * volume_fus / wing_area / span / l_f
        )
        partials["data:aerodynamics:fuselage:Cn_beta", "data:geometry:wing:area"] = (
            1.3 * volume_fus / wing_area ** 2.0 / span * (l_f / width_max)
        )
        partials["data:aerodynamics:fuselage:Cn_beta", "data:geometry:wing:span"] = (
            1.3 * volume_fus / wing_area / span ** 2.0 * (l_f / width_max)
        )
        partials["data:aerodynamics:fuselage:Cn_beta", "data:geometry:fuselage:volume"] = (
            -1.3 / wing_area / span * (l_f / width_max)
        )
