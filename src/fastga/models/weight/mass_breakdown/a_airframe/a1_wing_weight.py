"""Statistical estimation of wing weight."""
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
from .constants import SUBMODEL_WING_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_WING_MASS
] = "fastga.submodel.weight.mass.airframe.wing.legacy"


@oad.RegisterSubmodel(SUBMODEL_WING_MASS, "fastga.submodel.weight.mass.airframe.wing.legacy")
class ComputeWingWeight(om.ExplicitComponent):
    """
    Wing weight estimation

    Based on a statistical analysis. See :cite:`nicolai:2010` but can also be found in
    :cite:`gudmundsson:2013`.
    """

    def setup(self):

        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:wing:k_factor", val=1.0)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")

        self.add_output("data:weight:airframe:wing:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        wing_area = inputs["data:geometry:wing:area"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]

        a1 = (
            96.948
            * (
                (mtow * sizing_factor_ultimate / 10.0 ** 5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
        )  # mass formula in lb

        outputs["data:weight:airframe:wing:mass"] = (
            a1 * inputs["data:weight:airframe:wing:k_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        wing_area = inputs["data:geometry:wing:area"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]
        k_factor = inputs["data:weight:airframe:wing:k_factor"]

        a1 = (
            96.948
            * (
                (mtow * sizing_factor_ultimate / 10.0 ** 5.0) ** 0.65
                * (aspect_ratio / (np.cos(sweep_25) ** 2.0)) ** 0.57
                * (wing_area / 100.0) ** 0.61
                * ((1.0 + taper_ratio) / (2.0 * thickness_ratio)) ** 0.36
                * (1.0 + v_max_sl / 500.0) ** 0.5
            )
            ** 0.993
        )

        tmp = (
            np.sqrt(v_max_sl / 500 + 1)
            * (wing_area / 100) ** (61 / 100)
            * (aspect_ratio / np.cos(sweep_25) ** 2) ** (57 / 100)
            * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (9 / 25)
            * ((mtow * sizing_factor_ultimate) / 100000) ** (13 / 20)
        ) ** (7 / 1000)

        partials[
            "data:weight:airframe:wing:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = (
            (
                312875433
                * mtow
                * np.sqrt(v_max_sl / 500 + 1)
                * (wing_area / 100) ** (61 / 100)
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (57 / 100)
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (9 / 25)
            )
            / (500000000000 * ((mtow * sizing_factor_ultimate) / 100000) ** (7 / 20) * tmp)
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:area"] = (
            (
                1468107801
                * np.sqrt(v_max_sl / 500 + 1)
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (57 / 100)
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (9 / 25)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (13 / 20)
            )
            / (2500000000 * (wing_area / 100) ** (39 / 100) * tmp)
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:taper_ratio"] = (
            (
                216606069
                * np.sqrt(v_max_sl / 500 + 1)
                * (wing_area / 100) ** (61 / 100)
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (57 / 100)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (13 / 20)
            )
            / (
                12500000
                * thickness_ratio
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (16 / 25)
                * tmp
            )
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:thickness_ratio"] = (
            -(
                216606069
                * np.sqrt(v_max_sl / 500 + 1)
                * (taper_ratio + 1)
                * (wing_area / 100) ** (61 / 100)
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (57 / 100)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (13 / 20)
            )
            / (
                12500000
                * thickness_ratio ** 2
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (16 / 25)
                * tmp
            )
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:weight:aircraft:MTOW"] = (
            (
                312875433
                * sizing_factor_ultimate
                * np.sqrt(v_max_sl / 500 + 1)
                * (wing_area / 100) ** (61 / 100)
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (57 / 100)
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (9 / 25)
            )
            / (500000000000 * ((mtow * sizing_factor_ultimate) / 100000) ** (7 / 20) * tmp)
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:aspect_ratio"] = (
            (
                1371838437
                * np.sqrt(v_max_sl / 500 + 1)
                * (wing_area / 100) ** (61 / 100)
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (9 / 25)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (13 / 20)
            )
            / (
                25000000
                * np.cos(sweep_25) ** 2
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (43 / 100)
                * tmp
            )
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:sweep_25"] = (
            (
                1371838437
                * aspect_ratio
                * np.sin(sweep_25)
                * np.sqrt(v_max_sl / 500 + 1)
                * (wing_area / 100) ** (61 / 100)
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (9 / 25)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (13 / 20)
            )
            / (
                12500000
                * np.cos(sweep_25) ** 3
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (43 / 100)
                * tmp
            )
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:TLAR:v_max_sl"] = (
            (
                24067341
                * (wing_area / 100) ** (61 / 100)
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** (57 / 100)
                * ((taper_ratio + 1) / (2 * thickness_ratio)) ** (9 / 25)
                * ((mtow * sizing_factor_ultimate) / 100000) ** (13 / 20)
            )
            / (250000000 * np.sqrt(v_max_sl / 500 + 1) * tmp)
            * k_factor
        )
        partials["data:weight:airframe:wing:mass", "data:weight:airframe:wing:k_factor"] = a1
