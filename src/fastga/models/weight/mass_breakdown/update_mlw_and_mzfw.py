"""
Main component for mass breakdown.
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


class UpdateMLWandMZFW(ExplicitComponent):
    """
    Computes Maximum Landing Weight and Maximum Zero Fuel Weight from
    Overall Empty Weight and Maximum Payload.
    """

    def setup(self):
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")
        self.add_input("settings:weight:aircraft:MLW_MZFW_ratio", val=1.06)

        self.add_output("data:weight:aircraft:MZFW", units="kg")
        self.declare_partials("data:weight:aircraft:MZFW", "data:weight:aircraft:OWE", val=1.0)
        self.declare_partials(
            "data:weight:aircraft:MZFW", "data:weight:aircraft:max_payload", val=1.0
        )

        self.add_output("data:weight:aircraft:ZFW", units="kg")
        self.declare_partials("data:weight:aircraft:ZFW", "data:weight:aircraft:OWE", val=1.0)
        self.declare_partials("data:weight:aircraft:ZFW", "data:weight:aircraft:payload", val=1.0)

        self.add_output("data:weight:aircraft:MLW", units="kg")
        self.declare_partials(
            "data:weight:aircraft:MLW",
            [
                "data:weight:aircraft:MTOW",
                "data:weight:aircraft:OWE",
                "data:weight:aircraft:max_payload",
                "settings:weight:aircraft:MLW_MZFW_ratio",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        owe = inputs["data:weight:aircraft:OWE"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        max_pl = inputs["data:weight:aircraft:max_payload"]
        pl = inputs["data:weight:aircraft:payload"]
        cruise_ktas = inputs["data:TLAR:v_cruise"]

        mzfw = owe + max_pl
        zfw = owe + pl

        if cruise_ktas > 250.0:
            mlw = inputs["settings:weight:aircraft:MLW_MZFW_ratio"] * mzfw
        else:
            mlw = mtow

        outputs["data:weight:aircraft:MZFW"] = mzfw
        outputs["data:weight:aircraft:ZFW"] = zfw
        outputs["data:weight:aircraft:MLW"] = mlw

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        owe = inputs["data:weight:aircraft:OWE"]
        max_pl = inputs["data:weight:aircraft:max_payload"]
        cruise_ktas = inputs["data:TLAR:v_cruise"]

        mzfw = owe + max_pl

        if cruise_ktas > 250.0:
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MTOW",
            ] = 0.0
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:OWE",
            ] = inputs["settings:weight:aircraft:MLW_MZFW_ratio"]
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:max_payload",
            ] = inputs["settings:weight:aircraft:MLW_MZFW_ratio"]
            partials[
                "data:weight:aircraft:MLW",
                "settings:weight:aircraft:MLW_MZFW_ratio",
            ] = mzfw
        else:
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MTOW",
            ] = 1.0
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:OWE",
            ] = 0.0
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:max_payload",
            ] = 0.0
            partials[
                "data:weight:aircraft:MLW",
                "settings:weight:aircraft:MLW_MZFW_ratio",
            ] = 0.0
