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
    SUBMODEL_HORIZONTAL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.horizontal_tail.gd"
)
class ComputeHorizontalTailWeightGD(om.ExplicitComponent):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft.
    """

    def setup(self):

        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="ft")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="ft")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="ft")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="ft"
        )

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        span_ht = inputs["data:geometry:horizontal_tail:span"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        root_chord_ht = inputs["data:geometry:horizontal_tail:root:chord"]
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        root_thickness = t_c_ht * root_chord_ht

        a31 = (
            0.0034
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht ** 0.584
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
        )
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        span_ht = inputs["data:geometry:horizontal_tail:span"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        root_chord_ht = inputs["data:geometry:horizontal_tail:root:chord"]
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        k_factor = inputs["data:weight:airframe:horizontal_tail:k_factor"]

        root_thickness = t_c_ht * root_chord_ht

        a31 = (
            0.0034
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht ** 0.584
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
        )
        # Mass formula in lb

        tmp = (
            area_ht ** 0.5840
            * (mtow * sizing_factor_ultimate) ** 0.8130
            * (mac_ht / lp_ht) ** 0.2800
            * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
        ) ** 0.0850

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor * (
            (
                0.0025292
                * area_ht ** 0.5840
                * mtow
                * (mac_ht / lp_ht) ** 0.2800
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / ((mtow * sizing_factor_ultimate) ** 0.1870 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:weight:aircraft:MTOW"
        ] = k_factor * (
            (
                0.0025292
                * area_ht ** 0.5840
                * sizing_factor_ultimate
                * (mac_ht / lp_ht) ** 0.2800
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / ((mtow * sizing_factor_ultimate) ** 0.1870 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor * (
            (
                0.0018168
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / (area_ht ** 0.4160 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:span"
        ] = k_factor * (
            (
                1.0266e-4
                * area_ht ** 0.5840
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
            )
            / (root_chord_ht * t_c_ht * (span_ht / (root_chord_ht * t_c_ht)) ** 0.9670 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:thickness_ratio",
        ] = k_factor * (
            -(
                1.0266e-4
                * area_ht ** 0.5840
                * span_ht
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
            )
            / (root_chord_ht * t_c_ht ** 2 * (span_ht / (root_chord_ht * t_c_ht)) ** 0.9670 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:root:chord"
        ] = k_factor * (
            -(
                1.0266e-4
                * area_ht ** 0.5840
                * span_ht
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
            )
            / (root_chord_ht ** 2 * t_c_ht * (span_ht / (root_chord_ht * t_c_ht)) ** 0.9670 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:MAC:length"
        ] = k_factor * (
            (
                8.7108e-4
                * area_ht ** 0.5840
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / (lp_ht * (mac_ht / lp_ht) ** 0.7200 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor * (
            -(
                8.7108e-4
                * area_ht ** 0.5840
                * mac_ht
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / (lp_ht ** 2 * (mac_ht / lp_ht) ** 0.7200 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = a31
