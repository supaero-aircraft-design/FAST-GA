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

        self.declare_partials(of="*", wrt="*", method="exact")

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
            (0.01 * wing_area) ** 0.61
            * (0.002 * v_max_sl + 1.0) ** 0.5
            * ((0.5 * taper_ratio + 0.5) / thickness_ratio) ** 0.36
            * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.57
            * (1.0e-5 * mtow * sizing_factor_ultimate) ** 0.65
        ) ** 0.007

        partials[
            "data:weight:airframe:wing:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = (
            (
                6.2575e-4
                * mtow
                * (0.01 * wing_area) ** 0.61
                * (0.002 * v_max_sl + 1.0) ** 0.5
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.57
                * ((0.5 * (taper_ratio + 1.0)) / thickness_ratio) ** 0.36
            )
            / ((1.0e-5 * mtow * sizing_factor_ultimate) ** 0.35 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:area"] = (
            (
                0.58724
                * (0.002 * v_max_sl + 1.0) ** 0.5
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.57
                * ((0.5 * (taper_ratio + 1.0)) / thickness_ratio) ** 0.36
                * (1.0e-5 * mtow * sizing_factor_ultimate) ** 0.65
            )
            / ((0.01 * wing_area) ** 0.39 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:taper_ratio"] = (
            (
                17.328
                * (0.01 * wing_area) ** 0.61
                * (0.002 * v_max_sl + 1.0) ** 0.5
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.57
                * (1.0e-5 * mtow * sizing_factor_ultimate) ** 0.65
            )
            / (thickness_ratio * ((0.5 * taper_ratio + 0.5) / thickness_ratio) ** 0.64 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:thickness_ratio"] = (
            -(
                17.32848552
                * (taper_ratio + 1.0)
                * (0.01 * wing_area) ** 0.61
                * (0.002 * v_max_sl + 1.0) ** 0.5
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.57
                * (1.0e-5 * mtow * sizing_factor_ultimate) ** 0.65
            )
            / (thickness_ratio ** 2 * ((0.5 * taper_ratio + 0.5) / thickness_ratio) ** 0.64 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:weight:aircraft:MTOW"] = (
            (
                6.2575e-4
                * sizing_factor_ultimate
                * (0.01 * wing_area) ** 0.61
                * (0.002 * v_max_sl + 1.0) ** 0.5
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.57
                * ((0.5 * (taper_ratio + 1.0)) / thickness_ratio) ** 0.36
            )
            / ((1.0e-5 * mtow * sizing_factor_ultimate) ** 0.35 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:aspect_ratio"] = (
            (
                54.874
                * (0.01 * wing_area) ** 0.61
                * (0.002 * v_max_sl + 1.0) ** 0.5
                * ((0.5 * (taper_ratio + 1.0)) / thickness_ratio) ** 0.36
                * (1.0e-5 * mtow * sizing_factor_ultimate) ** 0.65
            )
            / (np.cos(sweep_25) ** 2 * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.43 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:geometry:wing:sweep_25"] = (
            (
                109.75
                * aspect_ratio
                * np.sin(sweep_25)
                * (0.01 * wing_area) ** 0.61
                * (0.002 * v_max_sl + 1.0) ** 0.5
                * ((0.5 * (taper_ratio + 1.0)) / thickness_ratio) ** 0.36
                * (1.0e-5 * mtow * sizing_factor_ultimate) ** 0.65
            )
            / (np.cos(sweep_25) ** 3 * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.43 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:TLAR:v_max_sl"] = (
            (
                0.096269
                * (0.01 * wing_area) ** 0.61
                * (aspect_ratio / np.cos(sweep_25) ** 2) ** 0.57
                * ((0.5 * (taper_ratio + 1.0)) / thickness_ratio) ** 0.36
                * (1.0e-5 * mtow * sizing_factor_ultimate) ** 0.65
            )
            / ((0.002 * v_max_sl + 1.0) ** 0.5 * tmp)
        ) * k_factor
        partials["data:weight:airframe:wing:mass", "data:weight:airframe:wing:k_factor"] = a1
