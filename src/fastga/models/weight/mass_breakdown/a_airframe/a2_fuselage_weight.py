"""Estimation of fuselage weight."""
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

import logging

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from .constants import SUBMODEL_FUSELAGE_MASS

_LOGGER = logging.getLogger(__name__)

oad.RegisterSubmodel.active_models[
    SUBMODEL_FUSELAGE_MASS
] = "fastga.submodel.weight.mass.airframe.fuselage.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_FUSELAGE_MASS, "fastga.submodel.weight.mass.airframe.fuselage.legacy"
)
class ComputeFuselageWeight(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on a statistical analysis. See :cite:`nicolai:2010` but can also be found in
    :cite:`gudmundsson:2013`.
    """

    def setup(self):

        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:fuselage:k_factor", val=1.0)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")

        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]

        a2 = (
            200.0
            * (
                (mtow * sizing_factor_ultimate / (10.0 ** 5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 3.28084
                / 10.0
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
        )  # mass formula in lb

        outputs["data:weight:airframe:fuselage:mass"] = (
            a2 * inputs["data:weight:airframe:fuselage:k_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]
        k_factor = inputs["data:weight:airframe:fuselage:k_factor"]

        a2 = (
            200.0
            * (
                (mtow * sizing_factor_ultimate / (10.0 ** 5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 3.28084
                / 10.0
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
        )  # mass formula in lb

        partials[
            "data:weight:airframe:fuselage:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor * (
            (
                129019033
                * mtow
                * (maximum_width + maximum_height)
                * ((82021 * fus_length) / 250000) ** (857 / 1000)
                * (v_max_sl / 100) ** (169 / 500)
                * (
                    (
                        82021
                        * (maximum_width + maximum_height)
                        * ((82021 * fus_length) / 250000) ** (857 / 1000)
                        * (v_max_sl / 100) ** (169 / 500)
                        * ((mtow * sizing_factor_ultimate) / 100000) ** (143 / 500)
                    )
                    / 250000
                )
                ** (1 / 10)
            )
            / (625000000000 * ((mtow * sizing_factor_ultimate) / 100000) ** (357 / 500))
        )
        partials["data:weight:airframe:fuselage:mass", "data:weight:aircraft:MTOW"] = k_factor * (
            (
                0.0002064
                * sizing_factor_ultimate
                * (0.3281 * fus_length) ** (857 / 1000)
                * (0.01 * v_max_sl) ** (169 / 500)
                * (maximum_width + maximum_height)
                * (
                    0.3281
                    * (0.3281 * fus_length) ** (857 / 1000)
                    * (0.01 * v_max_sl) ** (169 / 500)
                    * (maximum_width + maximum_height)
                    * (1.0e-5 * mtow * sizing_factor_ultimate) ** (143 / 500)
                )
                ** (1 / 10)
            )
            / (1.0e-5 * mtow * sizing_factor_ultimate) ** (357 / 500)
        )
        partials[
            "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_width"
        ] = k_factor * (
            (
                902231
                * ((82021 * fus_length) / 250000) ** (857 / 1000)
                * (v_max_sl / 100) ** (169 / 500)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (143 / 500)
                * (
                    (
                        82021
                        * (maximum_width + maximum_height)
                        * ((82021 * fus_length) / 250000) ** (857 / 1000)
                        * (v_max_sl / 100) ** (169 / 500)
                        * ((mtow * sizing_factor_ultimate) / 100000) ** (143 / 500)
                    )
                    / 250000
                )
                ** (1 / 10)
            )
            / 12500
        )
        partials[
            "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:maximum_height"
        ] = k_factor * (
            (
                902231
                * ((82021 * fus_length) / 250000) ** (857 / 1000)
                * (v_max_sl / 100) ** (169 / 500)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (143 / 500)
                * (
                    (
                        82021
                        * (maximum_width + maximum_height)
                        * ((82021 * fus_length) / 250000) ** (857 / 1000)
                        * (v_max_sl / 100) ** (169 / 500)
                        * ((mtow * sizing_factor_ultimate) / 100000) ** (143 / 500)
                    )
                    / 250000
                )
                ** (1 / 10)
            )
            / 12500
        )
        partials[
            "data:weight:airframe:fuselage:mass", "data:geometry:fuselage:length"
        ] = k_factor * (
            (
                63419618745307
                * (maximum_width + maximum_height)
                * (v_max_sl / 100) ** (169 / 500)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (143 / 500)
                * (
                    (
                        82021
                        * (maximum_width + maximum_height)
                        * ((82021 * fus_length) / 250000) ** (857 / 1000)
                        * (v_max_sl / 100) ** (169 / 500)
                        * ((mtow * sizing_factor_ultimate) / 100000) ** (143 / 500)
                    )
                    / 250000
                )
                ** (1 / 10)
            )
            / (3125000000000 * ((82021 * fus_length) / 250000) ** (143 / 1000))
        )
        partials["data:weight:airframe:fuselage:mass", "data:TLAR:v_max_sl"] = k_factor * (
            (
                0.244
                * (0.3281 * fus_length) ** (857 / 1000)
                * (maximum_width + maximum_height)
                * (1.0e-5 * mtow * sizing_factor_ultimate) ** (143 / 500)
                * (
                    0.3281
                    * (0.3281 * fus_length) ** (857 / 1000)
                    * (0.01 * v_max_sl) ** (169 / 500)
                    * (maximum_width + maximum_height)
                    * (1.0e-5 * mtow * sizing_factor_ultimate) ** (143 / 500)
                )
                ** (1 / 10)
            )
            / (0.01 * v_max_sl) ** (331 / 500)
        )
        partials[
            "data:weight:airframe:fuselage:mass", "data:weight:airframe:fuselage:k_factor"
        ] = a2
