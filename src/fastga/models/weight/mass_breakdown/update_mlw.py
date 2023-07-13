"""
Maximum Landing Weight (MLW) estimation.
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

from .constants import SUBMODEL_MLW


@oad.RegisterSubmodel(SUBMODEL_MLW, "fastga.submodel.weight.mass.mlw.legacy")
class ComputeMLW(om.ExplicitComponent):
    """
    Computes Maximum Landing Weight from Maximum Zero Fuel Weight.
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")
        self.add_input("settings:weight:aircraft:MLW_MZFW_ratio", val=1.06)
        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:MLW", units="kg")

        self.declare_partials(
            "data:weight:aircraft:MLW",
            [
                "data:weight:aircraft:MTOW",
                "settings:weight:aircraft:MLW_MZFW_ratio",
                "data:weight:aircraft:MZFW",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        cruise_ktas = inputs["data:TLAR:v_cruise"]
        mzfw = inputs["data:weight:aircraft:MZFW"]

        if cruise_ktas > 250.0:
            mlw = inputs["settings:weight:aircraft:MLW_MZFW_ratio"] * mzfw
        else:
            mlw = mtow

        outputs["data:weight:aircraft:MLW"] = mlw

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        cruise_ktas = inputs["data:TLAR:v_cruise"]
        mzfw = inputs["data:weight:aircraft:MZFW"]

        if cruise_ktas > 250.0:
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MTOW",
            ] = 0.0
            partials[
                "data:weight:aircraft:MLW",
                "settings:weight:aircraft:MLW_MZFW_ratio",
            ] = mzfw
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MZFW",
            ] = inputs["settings:weight:aircraft:MLW_MZFW_ratio"]
        else:
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MTOW",
            ] = 1.0
            partials[
                "data:weight:aircraft:MLW",
                "settings:weight:aircraft:MLW_MZFW_ratio",
            ] = 0.0
            partials[
                "data:weight:aircraft:MLW",
                "data:weight:aircraft:MZFW",
            ] = 0.0
