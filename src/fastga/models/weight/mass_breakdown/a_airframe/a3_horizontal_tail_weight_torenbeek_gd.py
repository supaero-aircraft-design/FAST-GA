"""
Estimation of tail weight.
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
import openmdao.api as om

import fastoad.api as oad

from .constants import SUBMODEL_HORIZONTAL_TAIL_MASS


@oad.RegisterSubmodel(
    SUBMODEL_HORIZONTAL_TAIL_MASS,
    "fastga.submodel.weight.mass.airframe.horizontal_tail.torenbeek_gd",
)
class ComputeHorizontalTailWeightTorenbeekGD(om.ExplicitComponent):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft. Should
    only be used with aircraft having a diving speed above 250 KEAS.
    """

    def setup(self):

        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", units="kn")

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        a31 = area_ht * (3.81 * area_ht ** 0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]
        k_factor = inputs["data:weight:airframe:horizontal_tail:k_factor"]

        a31 = area_ht * (3.81 * area_ht ** 0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)
        # Mass formula in lb

        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor * ((0.004572 * area_ht ** (0.2) * vd) / np.cos(sweep_25) - 0.287)
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:characteristic_speed:vd",
        ] = k_factor * ((0.00381 * area_ht ** (1.2)) / np.cos(sweep_25))
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:sweep_25"
        ] = k_factor * (
            (0.00381 * area_ht ** (1.2) * vd * np.sin(sweep_25)) / np.cos(sweep_25) ** 2
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = a31
